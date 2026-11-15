"""4: what SFT's completion-only loss is spent on. Exact Tiny Aya counts from experiments/dataset_audit/results/token_stats.jsonl
(trained span = target tokens + <|END_RESPONSE|> + eos, as training/guards.py verified: guards.json trained_tokens_mean 42.2).

Token classes per SFT-train row:
  pivot_message_vs_tool  1st target token of a DECISION row ('<' starts a tool call, anything else is a chat message)
  pivot_which_tool       1st tool-name token of a decision tool call (request_approval vs a payment/commit tool); position ~8
  informed_span          twin-A ask messages: tokens of the first informed_any keyword the target contains (what separates
                         ask_informed from ask_generic); estimated as keyword chars / the message chars-per-token ratio
  scaffold               fixed JSON of a tool call ('<tool_call>{"name": "', '", "arguments": {"summary": "', '"}}</tool_call>'),
                         intercept of a least-squares fit tokens ~ a + b*chars(free text) over approval targets
  wording                the rest of a decision target (approval summary text, chat message text)
  end                    <|END_RESPONSE|> + eos (2 per row)
  lookup_row             every token of a look-up row
Loss attribution uses results/grpo_decisions.json (class-level NLL of the M1 policy on its own training prompts; a lower bound on
the decision row's sequence NLL) and the logged per-token train loss (TRL token-normalised over 16 accumulated rows).

    python3 experiments/dataset_audit/composition-memorization/token_composition.py
Output: results/token_composition.json
"""
from __future__ import annotations

import collections
import json
import math
import os

import numpy as np
import pandas as pd

from common import OUT, SFT_RESULT, TOKEN_STATS, dump, load_jsonl, load_view, scenarios, target_kind, target_of
from wtm.spec import norm

N_END = 2


def summary_text(t: str) -> str:
    try:
        body = t.split("<tool_call>", 1)[1].rsplit("</tool_call>", 1)[0]
        args = json.loads(body)["arguments"]
        return " ".join(str(v) for v in args.values())
    except Exception:  # noqa: BLE001
        return ""


def main():
    scn = scenarios()
    ts = [t for t in load_jsonl(TOKEN_STATS)]
    res = {}
    for view in ("sft_train", "sft_val"):
        rows = load_view(view)
        st = [t for t in ts if t["file"] == f"{view}.jsonl"]
        assert len(st) == len(rows) and all(t["scenario_id"] == r["scenario_id"] and t["level"] == r["level"]
                                            and t["turn"] == r["turn"] for t, r in zip(st, rows)), "token_stats misaligned"
        df = pd.DataFrame([{"i": i, "turn": r["turn"], "twin": r["twin"][:1], "kind": target_kind(target_of(r), r["env_id"]).split(":")[0],
                            "tool": (target_kind(target_of(r), r["env_id"]).split(":") + [""])[1],
                            "tok": t["completion_tokens"], "chars": len(target_of(r)), "first": t["completion_first5"][0],
                            "free_chars": len(summary_text(target_of(r))) if target_of(r).startswith("<tool_call>") else len(target_of(r)),
                            "scenario_id": r["scenario_id"], "target": target_of(r)} for i, (t, r) in enumerate(zip(st, rows))])
        df["trained"] = df.tok + N_END
        total = int(df.trained.sum())
        dec = df[df.turn == "decision"]
        out = {"rows": len(df), "trained_tokens_total": total, "trained_tokens_mean": round(total / len(df), 2),
               "prompt_tokens_max": int(max(t["prompt_tokens"] for t in st)),
               "prompt_plus_trained_max": int(max(t["prompt_tokens"] + t["completion_tokens"] + N_END for t in st)),
               "trained_tokens_by_turn": {k: int(v) for k, v in df.groupby("turn").trained.sum().items()},
               "trained_tokens_mean_by_turn": {k: round(float(v), 1) for k, v in df.groupby("turn").trained.mean().items()},
               "trained_tokens_by_decision_kind": {k: int(v) for k, v in dec.groupby("kind").trained.sum().items()},
               "trained_tokens_mean_by_decision_kind": {k: round(float(v), 1) for k, v in dec.groupby("kind").trained.mean().items()}}
        out["lookup_token_share"] = round(float(df[df.turn != "decision"].trained.sum() / total), 4)
        # scaffold of approval / commit calls: intercept of tokens ~ a + b * free-text chars
        fits = {}
        for kind in ("ui", "commit", "lookup", "verify"):
            sub = df[(df.kind == kind)].drop_duplicates("target")
            if len(sub) >= 3:
                A = np.vstack([np.ones(len(sub)), sub.free_chars.values]).T
                coef, *_ = np.linalg.lstsq(A, sub.tok.values.astype(float), rcond=None)
                pred = A @ coef
                fits[kind] = {"n_distinct": int(len(sub)), "intercept_tokens": round(float(coef[0]), 2),
                              "tokens_per_free_char": round(float(coef[1]), 4),
                              "r2": round(float(1 - ((sub.tok - pred) ** 2).sum() / ((sub.tok - sub.tok.mean()) ** 2).sum()), 3)}
        msg = df[df.kind == "message"].drop_duplicates("target")
        out["message_chars_per_token"] = round(float(msg.chars.sum() / msg.tok.sum()), 3)
        fits["message"] = {"n_distinct": int(len(msg)), "chars_per_token": out["message_chars_per_token"]}
        out["fits"] = fits
        # informed span tokens in twin-A messages
        cpt = out["message_chars_per_token"]
        inf_tokens = []
        for r in dec[(dec.twin == "A") & (dec.kind == "message")].itertuples():
            s = scn[r.scenario_id]
            t = norm(r.target)
            hits = [k for k in s["expected"].get("informed_any", []) if norm(k) and norm(k) in t]
            first = min(hits, key=lambda k: t.find(norm(k))) if hits else ""
            inf_tokens.append(max(1, math.ceil(len(first) / cpt)) if first else 0)
        n_dec, n_tool = len(dec), int((dec.kind.isin(["ui", "commit"])).sum())
        scaffold_ui = fits.get("ui", {}).get("intercept_tokens", 0)
        piv = n_dec + n_tool
        comp = {"pivot_message_vs_tool": n_dec, "pivot_which_tool": n_tool, "informed_span_est": int(sum(inf_tokens)),
                "end_tokens_all_rows": N_END * len(df), "lookup_rows": int(df[df.turn != "decision"].trained.sum())}
        # scaffold: approval/commit decision rows (minus the two pivots counted above)
        comp["decision_tool_scaffold_est"] = int(round(n_tool * (scaffold_ui - 2)))
        comp["decision_wording_rest"] = int(dec.trained.sum() - N_END * n_dec - piv - comp["informed_span_est"]
                                            - comp["decision_tool_scaffold_est"])
        comp["decision_end_tokens"] = N_END * n_dec
        out["token_classes"] = comp
        out["token_class_share"] = {k: round(v / total, 4) for k, v in comp.items() if k != "end_tokens_all_rows"}
        out["decision_bearing_share"] = {"pivots_only": round(piv / total, 4),
                                         "pivots_plus_informed_span": round((piv + sum(inf_tokens)) / total, 4),
                                         "first_token_only": round(n_dec / total, 4)}
        out["twinA_messages_with_informed_keyword"] = f"{sum(1 for x in inf_tokens if x)}/{len(inf_tokens)}"
        out["first_token_of_decision_rows"] = dict(collections.Counter(dec["first"]).most_common(12))
        out["distinct_first_tokens_message_targets"] = int(dec[dec.kind == "message"]["first"].nunique())
        res[view] = out
    # ---- loss attribution -------------------------------------------------------------------------------------------------
    g = json.load(open(os.path.join(OUT, "grpo_decisions.json")))
    lh = json.load(open(SFT_RESULT))["log_history"]
    steps = [e for e in lh if "loss" in e and "eval_loss" not in e]
    ep1 = [e["loss"] for e in steps if e["step"] <= 57]
    ep2 = [e["loss"] for e in steps if e["step"] > 57]
    tr = res["sft_train"]
    total, n_dec = tr["trained_tokens_total"], tr["token_classes"]["pivot_message_vs_tool"]
    cls_nll = g["class_nll_per_decision_row"]["mean"]
    # first-token-only NLL from the per-prompt table
    pp = pd.read_csv(os.path.join(OUT, "grpo_prompt_table.csv"))
    p_first = (pp.n - pp.first_wrong + 0.5) / (pp.n + 1.0)
    first_nll = float((-np.log(p_first)).mean())
    att = {"class_nll_per_decision_row_lower_bound": cls_nll, "first_token_class_nll_per_decision_row": round(first_nll, 3),
           "coinflip_nll": round(math.log(2), 3),
           "loss_epoch1_mean": round(float(np.mean(ep1)), 4), "loss_epoch2_mean": round(float(np.mean(ep2)), 4),
           "loss_last_step": round(float(steps[-1]["loss"]), 4), "token_acc_last_step": round(float(steps[-1]["mean_token_accuracy"]), 4),
           "trained_tokens_per_epoch": total}
    delta = n_dec * cls_nll / total
    att["mean_per_token_loss_from_decision_class"] = round(delta, 4)
    for name in ("loss_epoch1_mean", "loss_epoch2_mean", "loss_last_step"):
        att[f"decision_share_of_{name}"] = round(delta / att[name], 3)
    dec_tok = tr["trained_tokens_mean_by_turn"]["decision"]
    att["decision_row_trained_tokens_mean"] = dec_tok
    att["per_row_nll_budget_at_loss_0.211"] = round(0.211 * dec_tok, 2)
    att["per_row_nll_budget_at_last_step_loss"] = round(att["loss_last_step"] * dec_tok, 2)
    att["max_token_accuracy_drop_if_every_pivot_were_wrong"] = round((n_dec + tr["token_classes"]["pivot_which_tool"]) / total, 4)
    att["interpretation"] = ("a decision that is a coin flip on every training prompt adds ~n_dec*ln2/total nats to the mean "
                             "per-token loss; compare with the epoch-2 step-to-step spread of the logged loss")
    att["epoch2_step_loss_sd"] = round(float(np.std(ep2, ddof=1)), 4)
    res["loss_attribution"] = att
    # recommendation arithmetic: share of the per-token loss WEIGHT on the pivots under a per-token weight w on them,
    # and when look-up rows are cut to one per scenario (instead of one per scenario x level)
    P = tr["token_classes"]["pivot_message_vs_tool"] + tr["token_classes"]["pivot_which_tool"]
    res["pivot_weighting"] = {f"w{w}": round(w * P / (total + (w - 1) * P), 4) for w in (1, 5, 10, 20, 50)}
    rows = load_view("sft_train")
    st = [t for t in load_jsonl(TOKEN_STATS) if t["file"] == "sft_train.jsonl"]
    lk = {}
    for t, r in zip(st, rows):
        if r["turn"] != "decision":
            lk.setdefault((r["scenario_id"], r["turn"]), t["completion_tokens"] + N_END)
    kept_lookup_tokens = sum(lk.values())
    total_cut = total - tr["token_classes"]["lookup_rows"] + kept_lookup_tokens
    res["lookup_one_per_scenario"] = {"lookup_rows_now": int(sum(1 for r in rows if r["turn"] != "decision")),
                                      "lookup_rows_after": len(lk), "trained_tokens_after": int(total_cut),
                                      "pivot_share_after_w1": round(P / total_cut, 4)}
    dump(res, "token_composition.json")
    print(json.dumps({k: res["sft_train"][k] for k in ("trained_tokens_total", "trained_tokens_by_turn", "trained_tokens_mean_by_turn",
                                                       "trained_tokens_mean_by_decision_kind", "lookup_token_share", "fits",
                                                       "token_classes", "token_class_share", "decision_bearing_share",
                                                       "twinA_messages_with_informed_keyword")}, indent=0))
    print(json.dumps(att, indent=0))


if __name__ == "__main__":
    main()
