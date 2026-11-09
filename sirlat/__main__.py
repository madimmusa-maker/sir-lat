"""Командная строка: заметки -> сценарий -> озвучка -> видео.

Примеры:
    python -m sirlat --notes notes/example.txt --format long
    python -m sirlat --notes notes/example.txt --format short
    python -m sirlat --notes notes/example.txt --format both --minutes 15
    python -m sirlat --script output/example/script_long.txt --format long
    python -m sirlat --notes notes/example.txt --offline --preview 20   # быстрая проверка без сети
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from .config import OUTPUT
from .render import LONG, SHORT, concat, part_duration, render_part, speak_part
from .script import Part, make_script, parse_script

SHORT_LIMIT = 58.5   # шортс должен быть короче 59 секунд
LONG_MIN, LONG_MAX = 10 * 60, 30 * 60


def _fmt(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def build(fmt: str, notes: str, out_dir: Path, args) -> Path:
    layout = LONG if fmt == "long" else SHORT
    script_file = out_dir / f"script_{fmt}.txt"
    own = getattr(args, "own", None)
    if own is not None:
        return build_own(fmt, own, out_dir, args, layout, script_file)
    if args.script:
        text = Path(args.script).read_text(encoding="utf-8")
    elif script_file.exists() and args.reuse_script:
        text = script_file.read_text(encoding="utf-8")
    else:
        text = make_script(notes, fmt, args.minutes, offline=args.offline)
    script_file.write_text(text, encoding="utf-8")
    script = parse_script(text)
    print(f"Сценарий «{script.title}»: {len(script.parts)} частей, {script.word_count()} слов → {script_file}")

    if fmt == "short":
        # один «кусок»: все фразы тизера
        phrases = [p for part in script.parts for p in part.phrases]
        script.parts = [Part(script.title, phrases)]

    print("Озвучка (Silero → edge-tts → gTTS)…")
    spoken_parts = [speak_part(p, args.offline) for p in script.parts]
    engines = [sp.engine for part in spoken_parts for _, sp in part]
    if len(set(engines)) > 1:
        # голос сменился посреди видео — переозвучиваем всё последним рабочим движком
        from collections import Counter
        final = engines[-1]
        print(f"Голос сменился по ходу озвучки ({dict(Counter(engines))}) — "
              f"переозвучиваю все фразы движком {final}")
        try:
            spoken_parts = [speak_part(p, args.offline, engine=final) for p in script.parts]
        except Exception as e:  # noqa: BLE001
            print(f"  не получилось ({type(e).__name__}: {e}) — оставляю смешанную озвучку")
    if engines:
        print(f"Озвучка: {engines[-1]}")

    if fmt == "short":
        sp = spoken_parts[0]
        while len(sp) > 2 and part_duration(sp) > SHORT_LIMIT:
            # убираем предпоследние фразы, сохраняя финальный призыв
            sp.pop(-2)
        spoken_parts[0] = sp
        script.parts[0].phrases = [ph for ph, _ in sp]

    total = sum(part_duration(s) for s in spoken_parts)
    print(f"Длительность по озвучке: {_fmt(total)}")
    if fmt == "long" and not args.preview:
        if total < LONG_MIN:
            print(f"ВНИМАНИЕ: видео короче 10 минут ({_fmt(total)}). "
                  "Увеличьте --minutes или добавьте больше заметок.")
        while total > LONG_MAX and len(spoken_parts) > 2:
            print("Видео длиннее 30 минут — убираю предпоследнюю часть.")
            spoken_parts.pop(-2)
            script.parts.pop(-2)
            total = sum(part_duration(s) for s in spoken_parts)

    return render_all(fmt, script, spoken_parts, out_dir, args, layout)


def build_own(fmt, own, out_dir: Path, args, layout, script_file: Path) -> Path:
    """Видео по своей записи голоса: сценарий и субтитры — из распознанного текста."""
    from .script import Phrase
    from .tts import Speech, Word
    script = parse_script(own.script_text(f"Хроники сэра Лата: {out_dir.name}"))
    items = []
    for text, emo, pose, audio, words in own.phrases:
        ph = Phrase(text, emo, pose)
        sp = Speech(audio, [Word(w["text"], w["start"], w["end"]) for w in words], "own")
        items.append((ph, sp))
    if fmt == "short":
        # тизер — начало записи, целыми фразами, до 59 секунд
        while len(items) > 1 and part_duration(items) > SHORT_LIMIT:
            items.pop()
        groups = [items]
    else:
        # длинное видео — части примерно по 2 минуты
        groups, cur = [], []
        for it in items:
            cur.append(it)
            if part_duration(cur) >= 120:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
    script.parts = [Part(f"Часть {i + 1}" if fmt == "long" else script.title, [ph for ph, _ in g])
                    for i, g in enumerate(groups)]
    script_file.write_text(
        "\n".join([f"# {script.title}"] + [
            line for part in script.parts for line in
            [f"## {part.title}"] + [f"[{_tag(ph)}] {ph.text}" for ph in part.phrases]]) + "\n",
        encoding="utf-8")
    print(f"Сценарий из записи: {sum(len(g) for g in groups)} фраз → {script_file}")
    total = sum(part_duration(g) for g in groups)
    print(f"Озвучка: свой голос, {_fmt(total)}")
    if fmt == "long" and total < LONG_MIN and not args.preview:
        print(f"ВНИМАНИЕ: видео короче 10 минут ({_fmt(total)}) — запись короткая.")
    return render_all(fmt, script, groups, out_dir, args, layout)


def _tag(ph) -> str:
    from .own_voice import RU_EMO, RU_POSE
    return RU_EMO.get(ph.emotion, "важный") + (f", {RU_POSE[ph.pose]}" if ph.pose in RU_POSE else "")


def render_all(fmt, script, spoken_parts, out_dir: Path, args, layout) -> Path:
    parts_dir = out_dir / f"parts_{fmt}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    files = []
    budget = args.preview
    for i, (part, spoken) in enumerate(zip(script.parts, spoken_parts)):
        if budget is not None and budget <= 0:
            break
        f = parts_dir / f"part_{i + 1:02d}.mp4"
        t0 = time.time()
        print(f"Рендер части {i + 1}/{len(script.parts)}: {part.title}")
        dur = render_part(part, spoken, layout, f, seed=i + 1,
                          show_title=(fmt == "long"), max_seconds=budget)
        print(f"  готово: {_fmt(dur)} за {time.time() - t0:.0f} с")
        files.append(f)
        if budget is not None:
            budget -= dur
    final = out_dir / f"{out_dir.name}_{fmt}.mp4"
    if len(files) == 1:
        shutil.copy(files[0], final)
    else:
        concat(files, final)
    if not args.keep_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"ГОТОВО: {final}")
    return final


def main(argv=None):
    ap = argparse.ArgumentParser(description="Хроники сэра Лата — генератор видео")
    ap.add_argument("--notes", help="файл с заметками (notes/*.txt)")
    ap.add_argument("--script", help="готовый размеченный сценарий (пропустить Gemini)")
    ap.add_argument("--format", choices=["long", "short", "both"], default="long")
    ap.add_argument("--minutes", type=float, default=float(os.getenv("SIRLAT_MINUTES", 14)),
                    help="целевая длина длинного видео в минутах (10–30)")
    ap.add_argument("--offline", action="store_true",
                    help="без Gemini и озвучки: тестовый сценарий и синтетический голос")
    ap.add_argument("--preview", type=float, default=None,
                    help="отрендерить только первые N секунд (для проверки)")
    ap.add_argument("--reuse-script", action="store_true",
                    help="взять уже сгенерированный сценарий из output/, если он есть")
    ap.add_argument("--keep-parts", action="store_true", help="не удалять отдельные части")
    ap.add_argument("--no-voice", action="store_true",
                    help="не использовать свою запись из voice/, даже если она есть")
    ap.add_argument("--out", default=None, help="папка для результатов")
    args = ap.parse_args(argv)

    if not args.notes and not args.script:
        ap.error("укажите --notes или --script")
    args.minutes = max(10.0, min(30.0, args.minutes))
    src = Path(args.notes or args.script)
    if not src.exists():
        ap.error(f"файл не найден: {src}")
    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else ""
    out_dir = Path(args.out) if args.out else OUTPUT / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # своя запись голоса: voice/<имя заметок>.m4a|mp3|…
    args.own = None
    if not args.no_voice:
        from . import own_voice
        rec = own_voice.find_recording(src.stem)
        if rec:
            try:
                args.own = own_voice.prepare(rec, out_dir, src.stem)
            except Exception as e:  # noqa: BLE001
                msg = f"Не удалось обработать запись {rec.name}: {type(e).__name__}: {e} — озвучу синтезом"
                print(f"::warning::{msg}" if os.getenv("GITHUB_ACTIONS") else f"ВНИМАНИЕ: {msg}")

    formats = ["long", "short"] if args.format == "both" else [args.format]
    for fmt in formats:
        build(fmt, notes, out_dir, args)


if __name__ == "__main__":
    sys.exit(main())
