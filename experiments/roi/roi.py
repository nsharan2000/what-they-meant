"""ROI: self-hosted (fine-tuned) Tiny Aya vs closed-model per-token billing, from MEASURED inputs only.

Inputs
  * closed models: mean cost_usd per decision episode from results rows (Claude Code list-price basis), realistic levels L1/L2
  * Tiny Aya: tokens per decision episode (rows' "tokens") and aggregate throughput = total tokens / wall-clock of the local vLLM run
    (first→last row timestamp, 32 concurrent episodes) on the DGX Spark GB10; L4 sensitivity = ×0.5 throughput
  * GPU price: HF Inference Endpoints NVIDIA L4 24 GB $0.80/h (experiments/roi/PRICING_SOURCES.md)
Volumes: 100k and 1M decisions / month. Self-host cost = max(always-on single L4 = 730 h × $0.80, hours needed × $0.80).
python3 experiments/roi/roi.py --closed <claude grid files> --local <local tiny aya grid> --out experiments/roi/roi.json [--since <epoch>]
(--since restricts the throughput measurement to one contiguous concurrent run; resumed runs contain idle gaps)
"""
import argparse
import glob
import json
from collections import defaultdict

L4_PER_HOUR = 0.80
HOURS_PER_MONTH = 730


def load(paths):
    return [json.loads(l) for p in paths for f in glob.glob(p) for l in open(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed", nargs="+", required=True)
    ap.add_argument("--local", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/roi/roi.json")
    ap.add_argument("--since", type=float, default=0.0, help="only local rows finished after this epoch time (one contiguous run)")
    ap.add_argument("--until", type=float, default=float("inf"), help="only local rows finished before this epoch time")
    ns = ap.parse_args()
    closed = [r for r in load(ns.closed) if r["model"].startswith("claude:") and r["level"] in ("L1", "L2", "L0r")]
    by = defaultdict(list)
    for r in closed:
        by[r["model"]].append(r.get("cost_usd", 0.0))
    closed_cost = {m: sum(v) / len(v) for m, v in by.items() if v}
    local = [r for r in load(ns.local) if r.get("tokens") and ns.since <= r["t"] <= ns.until]
    t0, t1 = min(r["t"] - r.get("seconds", 0) for r in local), max(r["t"] for r in local)
    tok = sum(r["tokens"]["prompt"] + r["tokens"]["completion"] for r in local)
    per_dec = tok / len(local)
    thr = tok / max(1.0, t1 - t0)
    out = {"closed_cost_per_decision_usd": {m: round(v, 5) for m, v in closed_cost.items()},
           "tiny_aya": {"episodes": len(local), "tokens_per_decision": round(per_dec), "throughput_tokens_per_s_gb10": round(thr),
                        "wall_clock_s": round(t1 - t0)}, "volumes": {}}
    for vol in (100_000, 1_000_000):
        hours_needed = vol * per_dec / thr / 3600
        row = {"self_host_L4_usd_month": round(max(HOURS_PER_MONTH, hours_needed) * L4_PER_HOUR, 2),
               "self_host_L4_usd_month_if_half_throughput": round(max(HOURS_PER_MONTH, 2 * hours_needed) * L4_PER_HOUR, 2),
               "gpu_hours_needed": round(hours_needed, 1)}
        for m, c in closed_cost.items():
            row[f"{m}_usd_month"] = round(vol * c, 2)
        out["volumes"][str(vol)] = row
    json.dump(out, open(ns.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
