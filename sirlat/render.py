"""Сборка видео: поляна с костром + сэр Лат + облачко с эмодзи + субтитры + голос.

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
from .campfire import CampfireScene, ground_shadow, light_sprite, log_sprite
from .overlays import bubble_events, bubble_state, game_window, thought_bubble
from .script import Part, Phrase
from .tts import Speech, Word, mouth_envelope, postprocess, synthesize


# ------------------------------------------------------------------ макеты ---

@dataclass
class Layout:
    name: str
    W: int
    H: int
    lat_h: int            # высота спрайта сэра Лата
    lat_cx: float         # центр сэра Лата по горизонтали (доля ширины кадра)
    sub_box: tuple        # (x0, y_center, x1) — зона субтитров
    sub_font: int
    sub_lines: int
    window_center: tuple  # центр игрового окна
    window_w: int
    title_y: int

    @property
    def lat_w(self):
        return round(character.W * self.lat_h / character.H)


# Сэр Лат сидит слева у бревна, у костра. Субтитры — там, где их не перекрывает
# ни он, ни костёр: в шортсе внизу (тёмный передний план), в длинном — справа.
LONG = Layout("long", 1920, 1080, lat_h=760, lat_cx=0.17,
              sub_box=(760, 1080 - 120, 1920 - 70), sub_font=56, sub_lines=2,
              window_center=(1240, 380), window_w=860, title_y=60)
SHORT = Layout("short", 1080, 1920, lat_h=700, lat_cx=0.21,
               sub_box=(60, 1690, 1020), sub_font=70, sub_lines=3,
               window_center=(540, 520), window_w=880, title_y=150)

LEAD_IN = 0.6      # тишина в начале части
TAIL = 0.7         # тишина в конце части
REVEAL_PAUSE = 0.8 # пауза перед развязкой


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
    pause: float = 0.3     # пауза перед фразой

    @property
    def end(self):
        return self.start + self.speech.duration


def speak_part(part: Part, offline: bool = False) -> list[tuple[Phrase, Speech]]:
    out = []
    for ph in part.phrases:
        out.append((ph, synthesize(ph.text, ph.emotion, offline=offline)))
    return out


def pause_before(prev: Phrase | None, ph: Phrase) -> float:
    """0.2–0.4 с между фразами; перед развязкой (или после интриги) — 0.8 с."""
    if prev is None:
        return 0.0
    if getattr(ph, "reveal", False) or prev.emotion == "intrigue":
        return REVEAL_PAUSE
    return random.Random(ph.text).uniform(0.2, 0.4)


def build_timeline(spoken):
    items, t, prev = [], LEAD_IN, None
    for ph, sp in spoken:
        p = pause_before(prev, ph)
        t += p
        items.append(Item(ph, sp, t, p))
        t += sp.duration
        prev = ph
    total = t + TAIL
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


def render_part(part: Part, spoken, layout: Layout, out_path: Path,
                seed: int = 0, show_title: bool = True, max_seconds: float | None = None,
                t0: float = 0.0):
    """t0 — время начала части в видео (чтобы зум и огонь шли без скачков между частями)."""
    items, total = build_timeline(spoken)
    if max_seconds:
        total = min(total, max_seconds)
    W, H = layout.W, layout.H
    n_frames = int(math.ceil(total * FPS))
    audio = postprocess(_mix_audio(items, total))     # лёгкое шумоподавление + −14 LUFS
    wav = out_path.with_suffix(".wav")
    _write_wav(wav, audio)
    env = mouth_envelope(audio, FPS, n_frames)

    scene = CampfireScene(W, H, FPS, seed=1)
    lat_anim = character.Animator(FPS, layout.lat_h, seed, screen_side="right", style="campfire")
    lat_w, lat_h = lat_anim.cw, lat_anim.ch
    lat_x = int(W * layout.lat_cx - lat_w / 2)
    lat_y = int(scene.ground_y + lat_h * 0.06 - lat_h)
    char_w = int(lat_w * character.BASE_W / character.W)
    fire_right = scene.fire_xy[0] > W * layout.lat_cx
    sh_rgb, sh_al = ground_shadow(int(char_w * 1.1), max(8, int(lat_h * 0.05)))
    log_rgb, log_al = log_sprite(int(char_w * 1.15), max(10, int(lat_h * 0.17)))
    log_x = int(W * layout.lat_cx - log_rgb.shape[1] / 2 - lat_w * 0.02)
    log_y = int(lat_y + lat_h - log_rgb.shape[0] * 0.78)
    # облачко над головой: справа сверху от шлема
    bsize = int(lat_w * 0.5)
    head_x = lat_x + (character.OFF_X + character.BASE_W * 0.8) * lat_h / character.H
    head_y = lat_y + (character.OFF_Y + character.BASE_H * 0.06) * lat_h / character.H
    events = bubble_events(items)

    # кому какая фраза соответствует в каждом кадре
    phrase_at = np.zeros(n_frames, np.int32)
    for idx, it in enumerate(items):
        a = int((it.start - (it.pause / 2 if idx else LEAD_IN)) * FPS)
        phrase_at[max(a, 0):] = idx

    sprites = SpriteCache()
    subs = SpriteCache(48)
    windows = SpriteCache()
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

    try:
        for fi in range(n_frames):
            t = fi / FPS
            # --- поляна: медленный зум, мерцание костра, искры
            frame, k = scene.frame(t0 + t)

            it_idx = int(phrase_at[fi])
            it = items[it_idx] if items else None

            # --- сэр Лат у бревна: тень на земле, свет костра, клюв в такт голосу
            blend(frame, sh_rgb, sh_al * (0.5 + 0.12 * (1 - k)),
                  int(W * layout.lat_cx - sh_rgb.shape[1] * (0.62 if fire_right else 0.38)),
                  int(scene.ground_y - sh_rgb.shape[0] / 2))
            speaking = bool(it) and it.start <= t <= it.end + 0.05
            vowel = None
            if speaking:
                rel = t - it.start
                for w in it.speech.words:
                    if w.start <= rel < w.end:
                        vowel = character.vowel_at(w.text, (rel - w.start) / max(1e-3, w.end - w.start))
                        break
            spr = lat_anim.calm_frame(t0 + t, float(env[fi]), vowel, speaking)
            rgb, al = _to_np_rgba(spr)
            blend(frame, light_sprite(rgb, al, k, fire_right), al, lat_x, lat_y)
            blend(frame, light_sprite(log_rgb, log_al, k * 0.9, fire_right), log_al, log_x, log_y)
            frame = scene.grade(frame)

            # --- облачко-мысль с эмодзи по эмоции фразы
            bs = bubble_state(events, t)
            if bs:
                emoji, op, sc = bs
                bimg = sprites.get(lambda: thought_bubble(emoji, bsize, True), ("bubble", emoji))
                brgb, bal = bimg
                if sc < 0.999:
                    pil = thought_bubble(emoji, bsize, True)
                    pil = pil.resize((max(1, int(pil.width * sc)), max(1, int(pil.height * sc))), Image.BILINEAR)
                    brgb, bal = _to_np_rgba(pil)
                bob = math.sin(2 * math.pi * t / 2.2) * lat_h * 0.004
                blend(frame, brgb, bal, int(head_x - brgb.shape[1] * 0.2),
                      int(head_y - brgb.shape[0] * 0.78 + bob), op)

            ph = it.phrase if it else Phrase("")
            # --- игровое окно
            if it and ph.window and it.start - 0.2 <= t <= it.end + 0.3:
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

            # --- субтитры по словам
            if it and it.start - 0.05 <= t <= it.end + 0.25:
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
        scene.close()
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
