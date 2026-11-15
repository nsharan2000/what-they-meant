"""Label adjudications (dataset/label_adjudications.json) applied on top of scenario specs — frozen copies stay untouched.

An adjudication only WIDENS what counts as correct for a row (expected.accept, e.g. ["approve", "act"] for a scheduled job that a
standing instruction covers); it never flips ask <-> proceed. Used by the grader (wtm.episode.decision_grade), the Opus judge
(wtm.judge) and the analyses (experiments/replication/analysis.py re-grades adjudicated rows from their stored decision class).
"""
from __future__ import annotations

import copy
import functools
import json
import os

from .spec import ROOT

PATH = os.path.join(ROOT, "dataset", "label_adjudications.json")


@functools.lru_cache(maxsize=1)
def adjudications() -> dict:
    if not os.path.exists(PATH):
        return {}
    return json.load(open(PATH)).get("rows", {})


def adjudicate(scn: dict) -> dict:
    """Return the scenario with its adjudication applied (a copy only when something changes)."""
    adj = adjudications().get(scn.get("id"))
    if not adj:
        return scn
    out = copy.deepcopy(scn)
    if adj.get("accept"):
        out["expected"]["accept"] = list(adj["accept"])
    return out


def accepted(scn: dict) -> set:
    exp = scn["expected"]
    return set(exp.get("accept") or [exp["decision"]])
