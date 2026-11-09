"""Стоковые видеоклипы с Pixabay (секрет PIXABAY_API_KEY).

Скачиваем короткие клипы по английскому запросу из сценария ({сток: ...}),
кэшируем в .cache/pixabay/. Если ключа нет или ничего не нашлось — фоном
будут только сгенерированные сцены.
"""
from __future__ import annotations

import hashlib
import json
import os
import random

import requests

from .config import CACHE

API = "https://pixabay.com/api/videos/"
FALLBACK_QUERIES = ["fantasy landscape", "medieval castle", "dark forest", "fire sparks",
                    "fog night", "magic particles", "stormy sky", "candle light"]


class Stock:
    def __init__(self, offline: bool = False):
        self.key = os.getenv("PIXABAY_API_KEY", "").strip()
        self.enabled = bool(self.key) and not offline
        self.dir = CACHE / "pixabay"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.used: set[int] = set()
        self._search_cache: dict[str, list[dict]] = {}
        if not self.enabled and not offline:
            print("ВНИМАНИЕ: PIXABAY_API_KEY не задан — стоковые клипы отключены.")

    def _search(self, query: str) -> list[dict]:
        if query in self._search_cache:
            return self._search_cache[query]
        f = self.dir / f"q_{hashlib.sha1(query.encode()).hexdigest()[:12]}.json"
        if f.exists():
            hits = json.loads(f.read_text())
        else:
            try:
                r = requests.get(API, params={
                    "key": self.key, "q": query, "per_page": 20,
                    "safesearch": "true", "video_type": "film",
                }, timeout=30)
                r.raise_for_status()
                hits = r.json().get("hits", [])
            except Exception as e:  # noqa: BLE001
                print(f"  Pixabay: ошибка поиска «{query}»: {e}")
                hits = []
            f.write_text(json.dumps(hits))
        self._search_cache[query] = hits
        return hits

    def clip(self, query: str | None, vertical: bool = False, seed: int = 0):
        """Возвращает путь к клипу (mp4) или None."""
        if not self.enabled:
            return None
        rnd = random.Random(seed)
        queries = [q for q in [query] if q] + rnd.sample(FALLBACK_QUERIES, 2)
        for q in queries:
            hits = [h for h in self._search(q) if h.get("duration", 0) >= 5]
            fresh = [h for h in hits if h["id"] not in self.used] or hits
            if not fresh:
                continue
            h = rnd.choice(fresh[:10])
            self.used.add(h["id"])
            vids = h.get("videos", {})
            v = vids.get("medium") or vids.get("large") or vids.get("small")
            if not v or not v.get("url"):
                continue
            path = self.dir / f"{h['id']}.mp4"
            if not path.exists():
                try:
                    with requests.get(v["url"], stream=True, timeout=120) as r:
                        r.raise_for_status()
                        tmp = path.with_suffix(".part")
                        with open(tmp, "wb") as fh:
                            for chunk in r.iter_content(1 << 20):
                                fh.write(chunk)
                        tmp.rename(path)
                except Exception as e:  # noqa: BLE001
                    print(f"  Pixabay: не скачался клип {h['id']}: {e}")
                    continue
            return path
        return None
