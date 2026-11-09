"""Сборка видео: фон + сэр Лат + субтитры + голос.

Каждая часть сценария рендерится в отдельный mp4 (кадры собираются в numpy и
подаются в ffmpeg через pipe), затем части склеиваются без перекодирования.
"""
from __future__ import annotations

import math
import random
import subprocess
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import character
from .config import FPS, SAMPLE_RATE, font
from .scenes import SCENE_KEYS, game_window, scene_image
from .script import Part, Phrase
from .stock import Stock
from .tts import Speech, Word, mouth_envelope, synthesize


# ------------------------------------------------------------------ макеты ---

@dataclass
class Layout:
    name: str
    W: int
    H: int
    lat_h: int            # высота спрайта сэра Лата
    sub_box: tuple        # (x0, y_center, x1) — зона субтитров
    sub_font: int
    sub_lines: int
    window_center: tuple  # центр игрового окна
    window_w: int
    title_y: int
    shot_len: float       # желаемая длина одного «плана» фона, сек

    @property
    def lat_w(self):
        return round(character.W * self.lat_h / character.H)

    @property
    def lat_xy(self):
        # правый нижний угол, низ брони слегка «обрезан» краем кадра
        return self.W - self.lat_w + int(self.lat_w * 0.04), self.H - self.lat_h + int(self.lat_h * 0.06)


LONG = Layout("long", 1920, 1080, lat_h=650,
              sub_box=(70, 1080 - 120, 1920 - 470 - 40), sub_font=56, sub_lines=2,
              window_center=(760, 430), window_w=900, title_y=60, shot_len=7.0)
SHORT = Layout("short", 1080, 1920, lat_h=720,
               sub_box=(60, 1110, 1020), sub_font=70, sub_lines=3,
               window_center=(540, 560), window_w=880, title_y=150, shot_len=4.0)

PAUSE = 0.38       # пауза между фразами
LEAD_IN = 0.6      # тишина в начале части
TAIL = 0.7         # тишина в конце части
XFADE = 0.4        # перекрёстная смена фона


# --------------------------------------------------------------- композитинг ---

def _to_np_rgba(img: Image.Image):
    a = np.asarray(img)
    return np.ascontiguousarray(a[..., :3]), a[..., 3:4].astype(np.float32) / 255.0


def blend(frame: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, x: int, y: int, opacity=1.0):
    H, W = frame.shape[:2]
    h, w = rgb.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    sx, sy = x0 - x, y0 - y
    a = alpha[sy:sy + y1 - y0, sx:sx + x1 - x0]
    if opacity != 1.0:
        a = a * opacity
    src = rgb[sy:sy + y1 - y0, sx:sx + x1 - x0].astype(np.float32)
    reg = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (reg + (src - reg) * a).astype(np.uint8)


class SpriteCache:
    """LRU-кэш готовых к наложению спрайтов (numpy)."""

    def __init__(self, maxsize=160):
        self.c = OrderedDict()
        self.maxsize = maxsize

    def get(self, img_fn, key):
        if key in self.c:
            self.c.move_to_end(key)
            return self.c[key]
        img = img_fn()
        val = _to_np_rgba(img) if img is not None else None
        self.c[key] = val
        if len(self.c) > self.maxsize:
            self.c.popitem(last=False)
        return val


# ------------------------------------------------------------------- фоны ---

class SceneBG:
    def __init__(self, key, seed, W, H, n_frames):
        self.img = scene_image(key, W, H, seed)
        self.W, self.H, self.n = W, H, max(1, n_frames)
        rnd = random.Random(seed)
        bh, bw = self.img.shape[:2]
        mx, my = bw - W, bh - H
        corners = [(0, 0), (mx, 0), (0, my), (mx, my), (mx // 2, my // 2)]
        self.p0 = rnd.choice(corners)
        self.p1 = rnd.choice([c for c in corners if c != self.p0])

    def frame(self, i):
        t = i / self.n
        t = t * t * (3 - 2 * t)
        x = int(self.p0[0] + (self.p1[0] - self.p0[0]) * t)
        y = int(self.p0[1] + (self.p1[1] - self.p0[1]) * t)
        return self.img[y:y + self.H, x:x + self.W].copy()

    def close(self):
        pass


class ClipBG:
    def __init__(self, path: Path, W, H, n_frames, fallback: SceneBG):
        self.W, self.H, self.fb = W, H, fallback
        self.size = W * H * 3
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},"
              "eq=brightness=-0.06:saturation=0.85")
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-stream_loop", "-1", "-i", str(path), "-an",
             "-vf", vf, "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            stdout=subprocess.PIPE)
        self.broken = False

    def frame(self, i):
        if not self.broken:
            buf = self.proc.stdout.read(self.size)
            if len(buf) == self.size:
                return np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 3).copy()
            self.broken = True
        return self.fb.frame(i)

    def close(self):
        try:
            self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass


# -------------------------------------------------------------- субтитры ---

def _paginate(words: list[Word], fnt, max_w: int, max_lines: int):
    """Делит слова фразы на «страницы» субтитров (до max_lines строк)."""
    pages, lines, line = [], [], []
    space = fnt.getlength(" ")

    def width(ws):
        return sum(fnt.getlength(w.text) for w in ws) + space * (len(ws) - 1)

    for idx, w in enumerate(words):
        if line and width(line + [w]) > max_w:
            carry = []
            # короткие предлоги/союзы («в», «и», «с»…) не оставляем в конце строки
            while len(line) > 1 and len(line[-1].text.strip(",.!?…:;—-")) <= 2:
                carry.insert(0, line.pop())
            lines.append(line)
            line = carry
            if len(lines) == max_lines:
                pages.append(lines)
                lines = []
        line.append(w)
    if line:
        lines.append(line)
    if lines:
        pages.append(lines)
    return pages


def _render_sub(lines, active_i: int, fnt, max_w):
    lh = int(fnt.size * 1.28)
    pad = int(fnt.size * 0.45)
    h = lh * len(lines) + pad * 2
    w = max_w + pad * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    space = fnt.getlength(" ")
    widest = max(sum(fnt.getlength(x.text) for x in ln) + space * (len(ln) - 1) for ln in lines)
    bx0 = (w - widest) / 2 - pad
    d.rounded_rectangle((bx0, 0, bx0 + widest + 2 * pad, h), radius=pad, fill=(0, 0, 0, 120))
    stroke = max(2, fnt.size // 14)
    y = pad
    wi = 0
    for ln in lines:
        lw = sum(fnt.getlength(x.text) for x in ln) + space * (len(ln) - 1)
        x = (w - lw) / 2
        for word in ln:
            col = (255, 214, 64) if wi == active_i else (255, 255, 255)
            wi += 1
            d.text((x, y + lh / 2), word.text, font=fnt, fill=col, anchor="lm",
                   stroke_width=stroke, stroke_fill=(10, 10, 16))
            x += fnt.getlength(word.text) + space
        y += lh
    return img


def _title_card(text: str, W: int, size: int):
    fnt = font(size, "serif")
    tw = fnt.getlength(text)
    pad = size // 2
    img = Image.new("RGBA", (int(tw + pad * 3), int(size * 1.9)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=pad,
                        fill=(20, 16, 30, 200), outline=(236, 186, 64, 255), width=4)
    d.text((img.width / 2, img.height / 2), text, font=fnt, fill=(255, 226, 150), anchor="mm",
           stroke_width=2, stroke_fill=(0, 0, 0))
    if img.width > W - 40:
        img = img.resize((W - 40, int(img.height * (W - 40) / img.width)), Image.LANCZOS)
    return img


def _brand(W):
    fnt = font(30, "serif")
    text = "ХРОНИКИ СЭРА ЛАТА"
    img = Image.new("RGBA", (int(fnt.getlength(text) + 40), 56), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((20, 28), text, font=fnt, fill=(255, 255, 255, 170), anchor="lm",
           stroke_width=2, stroke_fill=(0, 0, 0, 170))
    return img


# -------------------------------------------------------------- таймлайн ---

@dataclass
class Item:
    phrase: Phrase
    speech: Speech
    start: float

    @property
    def end(self):
        return self.start + self.speech.duration


def speak_part(part: Part, offline: bool) -> list[tuple[Phrase, Speech]]:
    out = []
    for ph in part.phrases:
        out.append((ph, synthesize(ph.text, offline=offline)))
    return out


def build_timeline(spoken):
    items, t = [], LEAD_IN
    for ph, sp in spoken:
        items.append(Item(ph, sp, t))
        t += sp.duration + PAUSE
    total = t - PAUSE + TAIL
    return items, total


def part_duration(spoken) -> float:
    return build_timeline(spoken)[1]


def _mix_audio(items, total):
    full = max([it.end for it in items] + [total]) + 1
    audio = np.zeros(int(math.ceil(full * SAMPLE_RATE)), np.float32)
    for it in items:
        a = int(it.start * SAMPLE_RATE)
        audio[a:a + len(it.speech.audio)] += it.speech.audio
    audio = audio[:int(total * SAMPLE_RATE)]
    peak = float(np.max(np.abs(audio))) if len(audio) else 1
    if peak > 0.98:
        audio *= 0.98 / peak
    return audio


def _write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _plan_shots(items, total, layout: Layout, stock: Stock, seed: int):
    """Разбивает часть на «планы» фона: сгенерированные сцены и стоковые клипы."""
    rnd = random.Random(seed)
    shots = []  # (start_time, kind, value)
    cur_start, cur_scene = 0.0, None
    k = 0
    default_cycle = ["dungeon", "city", "tavern", "hero", "monster"]
    for i, it in enumerate(items):
        ph = it.phrase
        scene_changed = ph.scene and ph.scene != cur_scene
        long_enough = it.start - cur_start >= layout.shot_len
        if i == 0 or scene_changed or long_enough:
            start = 0.0 if i == 0 else it.start - PAUSE / 2
            scene = ph.scene or cur_scene or default_cycle[seed % len(default_cycle)]
            use_stock = stock.enabled and not scene_changed and k % 2 == 1 and scene != "game"
            if use_stock:
                shots.append([start, "stock", ph.stock, scene])
            else:
                if not scene_changed and i > 0 and not ph.scene and rnd.random() < 0.35:
                    scene = rnd.choice(SCENE_KEYS[:5])
                shots.append([start, "scene", scene, scene])
            cur_start, cur_scene = it.start, ph.scene or cur_scene or scene
            k += 1
    for j, s in enumerate(shots):
        s.append(shots[j + 1][0] if j + 1 < len(shots) else total)
    return shots  # [start, kind, value, scene, end]


# ------------------------------------------------------------- мимика ---

STRONG_WORDS = {"никогда", "всегда", "невероятно", "ужас", "серьёзно", "внимание", "клянусь",
                "абсолютно", "огромный", "огромная", "вообще", "честно", "именно", "главное"}


def _is_emphatic(ph: Phrase) -> bool:
    """«Важная» фраза: восклицание или важный/злой тон — сэр Лат подаётся вперёд."""
    return "!" in ph.text or ph.emotion in ("important", "angry") and len(ph.text.split()) >= 5


def _gestures(items, env, seed):
    """Ударные слова -> короткие взмахи крылом: (время, крыло, сила в градусах)."""
    rnd = random.Random(seed * 17 + 3)
    out, last = [], -10.0
    for it in items:
        words = it.speech.words
        if not words:
            continue
        loud = []
        for w in words:
            a, b = int((it.start + w.start) * FPS), int((it.start + w.end) * FPS) + 1
            seg = env[a:b]
            loud.append(float(seg.mean()) if len(seg) else 0.0)
        mean = (sum(loud) / len(loud)) or 1.0
        for w, lv in zip(words, loud):
            core = w.text.strip(".,!?…:;—«»\"'()").lower()
            score = lv / mean
            score += 0.7 if w.text.rstrip("»\")").endswith("!") else 0
            score += 0.5 if core in STRONG_WORDS else 0
            score += 0.3 if len(core) >= 9 else 0
            score += 0.3 if any(ch.isdigit() for ch in core) else 0
            t = it.start + w.start
            if score >= 1.55 and t - last >= 1.6:
                side = "l" if rnd.random() < 0.7 else "r"   # чаще — крылом к экрану
                out.append((t - 0.05, side, rnd.uniform(14, 22)))
                last = t
    return out


# ----------------------------------------------------------------- рендер ---

def render_part(part: Part, spoken, layout: Layout, out_path: Path, stock: Stock,
                seed: int = 0, show_title: bool = True, max_seconds: float | None = None):
    items, total = build_timeline(spoken)
    if max_seconds:
        total = min(total, max_seconds)
    W, H = layout.W, layout.H
    n_frames = int(math.ceil(total * FPS))
    audio = _mix_audio(items, total)
    wav = out_path.with_suffix(".wav")
    _write_wav(wav, audio)
    env = mouth_envelope(audio, FPS, n_frames)

    gestures = _gestures(items, env, seed)
    emphatic = [_is_emphatic(it.phrase) for it in items]
    lat_anim = character.Animator(FPS, layout.lat_h, seed, gestures)
    # кому какая фраза соответствует в каждом кадре
    phrase_at = np.zeros(n_frames, np.int32)
    for idx, it in enumerate(items):
        a = int((it.start - (PAUSE / 2 if idx else LEAD_IN)) * FPS)
        phrase_at[max(a, 0):] = idx

    shots = _plan_shots(items, total, layout, stock, seed)
    sprites = SpriteCache()
    subs = SpriteCache(48)
    windows = SpriteCache()
    lat_x, lat_y = layout.lat_xy
    sx0, sy, sx1 = layout.sub_box
    sub_font = font(layout.sub_font)
    sub_w = sx1 - sx0 - int(layout.sub_font * 0.9)
    pages_cache = {}
    title = sprites.get(lambda: _title_card(part.title, W, 46 if layout.name == "long" else 52), "title")
    brand = sprites.get(lambda: _brand(W), "brand")

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0", "-i", str(wav),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
           "-g", str(FPS * 4), "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
           "-shortest", "-movflags", "+faststart", str(out_path)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    shot_idx, bg, prev_last = -1, None, None
    try:
        for fi in range(n_frames):
            t = fi / FPS
            # --- фон
            while shot_idx + 1 < len(shots) and t >= shots[shot_idx + 1][0]:
                if bg is not None:
                    bg.close()
                shot_idx += 1
                s0, kind, val, scene, s1 = shots[shot_idx]
                nf = int((s1 - s0) * FPS) + 2
                sc = SceneBG(scene, seed * 31 + shot_idx, W, H, nf)
                bg = sc
                if kind == "stock":
                    clip = stock.clip(val, vertical=layout.name == "short", seed=seed * 131 + shot_idx)
                    if clip:
                        bg = ClipBG(clip, W, H, nf, sc)
                shot_start_frame = fi
            local = fi - shot_start_frame
            frame = bg.frame(local)
            if prev_last is not None and local < XFADE * FPS and shot_idx > 0:
                k = local / (XFADE * FPS)
                frame = (prev_last * (1 - k) + frame * k).astype(np.uint8)
            # запомним последний кадр плана для перехода
            if shot_idx + 1 < len(shots) and t + 1 / FPS >= shots[shot_idx + 1][0]:
                prev_last = frame.astype(np.float32)

            it_idx = int(phrase_at[fi])
            it = items[it_idx] if items else None
            ph = it.phrase if it else Phrase("")

            # --- игровое окно
            if it and ph.window and it.start - 0.2 <= t <= it.end + PAUSE:
                rgbw = windows.get(lambda: game_window(ph.window, layout.window_w), ph.window)
                rgb, al = rgbw
                appear = min(1.0, (t - it.start + 0.2) / 0.25)
                cx, cy = layout.window_center
                blend(frame, rgb, al, int(cx - rgb.shape[1] / 2), int(cy - rgb.shape[0] / 2), appear)

            # --- заголовок части и логотип
            if show_title and t < 3.2:
                op = min(1.0, t / 0.3, (3.2 - t) / 0.4)
                rgb, al = title
                tx = 40 if layout.name == "long" else (W - rgb.shape[1]) // 2
                blend(frame, rgb, al, tx, layout.title_y, max(0.0, op))
            rgb, al = brand
            if layout.name == "long":
                blend(frame, rgb, al, W - rgb.shape[1] - 20, 16)
            else:
                blend(frame, rgb, al, (W - rgb.shape[1]) // 2, 40)

            # --- сэр Лат
            emo, pose = ph.emotion, ph.pose
            speaking = bool(it) and it.start <= t <= it.end + 0.05
            vowel = None
            if speaking:
                rel = t - it.start
                for w in it.speech.words:
                    if w.start <= rel < w.end:
                        vowel = character.vowel_at(w.text, (rel - w.start) / max(1e-3, w.end - w.start))
                        break
            spr = lat_anim.frame(t, emo, pose, float(env[fi]), vowel,
                                 emphatic=bool(it) and emphatic[it_idx] and it.start - 0.2 <= t <= it.end,
                                 speaking=speaking)
            rgb, al = _to_np_rgba(spr)
            blend(frame, rgb, al, lat_x, lat_y)

            # --- субтитры по словам
            if it and it.start - 0.05 <= t <= it.end + PAUSE * 0.6:
                if it_idx not in pages_cache:
                    pages_cache.clear()
                    pages_cache[it_idx] = _paginate(it.speech.words, sub_font, sub_w, layout.sub_lines)
                pages = pages_cache[it_idx]
                rel = t - it.start
                page_i = 0
                for pi, pg in enumerate(pages):
                    if rel >= pg[0][0].start - 0.05:
                        page_i = pi
                lines = pages[page_i]
                # подсвечено последнее начавшееся слово текущей страницы
                active_i = -1
                for wi, w in enumerate(x for ln in lines for x in ln):
                    if rel >= w.start - 0.02:
                        active_i = wi
                skey = (it_idx, page_i, active_i)
                rgb, al = subs.get(lambda: _render_sub(lines, active_i, sub_font, sub_w), skey)
                blend(frame, rgb, al, int((sx0 + sx1) / 2 - rgb.shape[1] / 2), int(sy - rgb.shape[0] / 2))

            enc.stdin.write(frame.tobytes())
    finally:
        if bg is not None:
            bg.close()
        enc.stdin.close()
        enc.wait()
        wav.unlink(missing_ok=True)
    if enc.returncode != 0:
        raise RuntimeError(f"ffmpeg завершился с ошибкой при сборке {out_path}")
    return total


def concat(parts: list[Path], out: Path):
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    lst.unlink(missing_ok=True)
