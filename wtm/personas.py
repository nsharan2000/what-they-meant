"""Persona sourcing from NVIDIA Nemotron-Personas (CC-BY-4.0, 1M synthetic personas per country, HF datasets).

Each member keeps the seed scenario's name (the user message refers to it) but borrows a demographically matching Nemotron persona
(occupation, city, persona text, cultural background) so profiles and spending histories are grounded in a standard, citable
source. Provenance (dataset id + uuid) is stored in member["profile"]["persona_source"].

    from wtm.personas import find_persona
    p = find_persona("USA", age=29, sex="Female", seed=1)      # -> dict or None
Datasets used: nvidia/Nemotron-Personas-{USA,India,UK,Germany,France,Japan,Brazil,Singapore,...}
India has splits en_IN / hi_Deva_IN / hi_Latn_IN (Hinglish, useful for noisy L3 messages).
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from .drivers.openai_compat import hf_token
from .spec import ROOT

API = "https://datasets-server.huggingface.co"
CACHE = os.path.join(ROOT, "envs", "_persona_cache")
SPLIT = {"India": "en_IN"}
KEEP = ["uuid", "persona", "professional_persona", "cultural_background", "sex", "age", "marital_status", "education_level",
        "occupation", "city", "state", "district", "zone", "country", "first_language", "second_language"]


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {hf_token()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _pool(country: str, n_batches: int = 5) -> list[dict]:
    """Fixed, cached pool of ~500 personas per country (rows endpoint at deterministic offsets; filter locally)."""
    ds = f"nvidia/Nemotron-Personas-{country}"
    split = SPLIT.get(country, "train")
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{country}_{split}_pool.jsonl")
    if os.path.exists(path):
        return [json.loads(l) for l in open(path) if l.strip()]
    rows = []
    for b in range(n_batches):
        off = 1000 + b * 197_000            # spread across the 1M rows, deterministic
        url = f"{API}/rows?dataset={urllib.parse.quote(ds)}&config=default&split={split}&offset={off}&length=100"
        for attempt in range(4):
            try:
                rows += [r["row"] for r in _get(url).get("rows", [])]
                break
            except Exception:  # noqa: BLE001 — datasets-server is occasionally flaky
                import time
                time.sleep(3 * (attempt + 1))
    with open(path, "w") as fh:
        for r in rows:
            out = {k: r.get(k) for k in KEEP if k in r}
            out["persona_source"] = {"dataset": ds, "split": split, "uuid": r.get("uuid"), "license": "CC-BY-4.0"}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    return _pool(country) if rows else []


def find_persona(country: str, age: int | None = None, sex: str | None = None, seed: int = 0) -> dict | None:
    """Deterministic: the `seed`-th closest-age persona of the requested sex from the country pool."""
    pool = _pool(country)
    if sex:
        pool = [p for p in pool if str(p.get("sex", "")).lower() == sex.lower()] or pool
    if age is not None:
        def gap(p):
            try:
                return abs(int(p.get("age")) - int(age))
            except (TypeError, ValueError):
                return 99
        pool = sorted(pool, key=lambda p: (gap(p), str(p.get("persona_source", {}).get("uuid"))))
    return pool[seed % len(pool)] if pool else None


if __name__ == "__main__":
    import sys
    print(json.dumps(find_persona(sys.argv[1] if len(sys.argv) > 1 else "USA", 29, "Female", 0), indent=1)[:1500])
