"""5: memorisation signatures in the M1 SFT log (experiments/training_eval/results/train_logs/m1_sft/result.json).

  * epoch boundary (57 optimizer steps per epoch: 904 rows / (1 x 16)): last-10 vs first-10 step losses, and the observed
    epoch-2 start against a log-linear trend fitted on the second half of epoch 1;
  * eval at epoch 1 vs 2: loss, token accuracy, entropy (confident-and-wrong = loss up, accuracy flat, entropy down);
  * exposure arithmetic: per epoch, how many rows repeat a target already seen EARLIER IN THE SAME EPOCH (exact string), and the
    expected share of such repeats per step under a random order (numpy simulation; the real order is TRL's seeded sampler,
    not reproducible without torch).

    python3 experiments/dataset_audit/composition-memorization/sft_log.py
Output: results/sft_log.json
"""
from __future__ import annotations

import json

import numpy as np

from common import SFT_RESULT, dump, load_view, target_of

STEPS_PER_EPOCH = 57


def main():
    lh = json.load(open(SFT_RESULT))["log_history"]
    st = [e for e in lh if "loss" in e and "eval_loss" not in e]
    ev = [e for e in lh if "eval_loss" in e]
    step = np.array([e["step"] for e in st])
    loss = np.array([e["loss"] for e in st])
    acc = np.array([e["mean_token_accuracy"] for e in st])
    ent = np.array([e["entropy"] for e in st])
    lr = np.array([e["learning_rate"] for e in st])
    ntok = np.diff(np.concatenate([[0.0], [e["num_tokens"] for e in st]]))
    out = {"n_steps": int(len(st)), "steps_per_epoch": STEPS_PER_EPOCH}
    e1, e2 = step <= STEPS_PER_EPOCH, step > STEPS_PER_EPOCH
    out["epoch_means"] = {"loss": [round(float(loss[e1].mean()), 4), round(float(loss[e2].mean()), 4)],
                          "token_acc": [round(float(acc[e1].mean()), 4), round(float(acc[e2].mean()), 4)],
                          "entropy": [round(float(ent[e1].mean()), 4), round(float(ent[e2].mean()), 4)]}
    for k in (5, 10):
        a = loss[(step > STEPS_PER_EPOCH - k) & e1]
        b = loss[(step > STEPS_PER_EPOCH) & (step <= STEPS_PER_EPOCH + k)]
        out[f"boundary_last{k}_vs_first{k}"] = {"epoch1_end_mean": round(float(a.mean()), 4), "epoch2_start_mean": round(float(b.mean()), 4),
                                                "ratio": round(float(b.mean() / a.mean()), 3),
                                                "epoch1_end_median": round(float(np.median(a)), 4),
                                                "epoch2_start_median": round(float(np.median(b)), 4)}
    # log-linear trend on the second half of epoch 1, extrapolated over the first 10 steps of epoch 2
    m = (step >= 29) & e1
    b1, b0 = np.polyfit(step[m], np.log(loss[m]), 1)
    nxt = (step > STEPS_PER_EPOCH) & (step <= STEPS_PER_EPOCH + 10)
    pred = np.exp(b0 + b1 * step[nxt])
    out["trend_epoch1_second_half"] = {"slope_log_loss_per_step": round(float(b1), 4),
                                       "predicted_epoch2_first10_mean": round(float(pred.mean()), 4),
                                       "observed_epoch2_first10_mean": round(float(loss[nxt].mean()), 4),
                                       "observed_over_predicted": round(float(loss[nxt].mean() / pred.mean()), 3)}
    m2 = e2
    c1, c0 = np.polyfit(step[m2], np.log(loss[m2]), 1)
    out["trend_epoch2"] = {"slope_log_loss_per_step": round(float(c1), 4)}
    # residual noise: correlation of the per-step loss with the per-step token count (row mix: L2 rows are long)
    out["corr_loss_vs_step_tokens_epoch2"] = round(float(np.corrcoef(loss[e2], ntok[e2])[0, 1]), 3)
    out["eval"] = [{"epoch": e["epoch"], "eval_loss": round(e["eval_loss"], 4), "eval_token_acc": round(e["eval_mean_token_accuracy"], 4),
                    "eval_entropy": round(e["eval_entropy"], 4)} for e in ev]
    out["train_eval_gap"] = {"epoch1_last10_train_mean": out["boundary_last10_vs_first10"]["epoch1_end_mean"],
                             "epoch1_eval": round(ev[0]["eval_loss"], 4),
                             "epoch2_last10_train_mean": round(float(loss[step > 2 * STEPS_PER_EPOCH - 10].mean()), 4),
                             "epoch2_eval": round(ev[1]["eval_loss"], 4)}
    out["eval_change"] = {"loss_rel": round(ev[1]["eval_loss"] / ev[0]["eval_loss"] - 1, 3),
                          "token_acc_abs": round(ev[1]["eval_mean_token_accuracy"] - ev[0]["eval_mean_token_accuracy"], 4),
                          "entropy_rel": round(ev[1]["eval_entropy"] / ev[0]["eval_entropy"] - 1, 3)}
    out["per_step"] = [{"step": int(s), "loss": round(float(l), 4), "acc": round(float(a), 4), "entropy": round(float(h), 4),
                        "lr": float(r), "tokens": int(t)} for s, l, a, h, r, t in zip(step, loss, acc, ent, lr, ntok)]
    # ---- exposure arithmetic ----------------------------------------------------------------------------------------------
    rows = load_view("sft_train")
    tg = [target_of(r) for r in rows]
    uniq = {t: k for k, t in enumerate(dict.fromkeys(tg))}
    ids = np.array([uniq[t] for t in tg])
    turn = np.array([r["turn"] == "decision" for r in rows])
    out["exposures"] = {"rows": len(rows), "distinct_targets": len(uniq),
                        "rows_repeating_a_target_seen_earlier_in_the_same_epoch": len(rows) - len(uniq),
                        "share": round(1 - len(uniq) / len(rows), 3),
                        "mean_exposures_per_distinct_target_2_epochs": round(2 * len(rows) / len(uniq), 2),
                        "mean_exposures_per_distinct_decision_target_2_epochs":
                            round(2 * int(turn.sum()) / len({t for t, d in zip(tg, turn) if d}), 2),
                        "mean_exposures_per_distinct_lookup_target_2_epochs":
                            round(2 * int((~turn).sum()) / len({t for t, d in zip(tg, turn) if not d}), 2)}
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(2000):
        perm = rng.permutation(len(ids))
        seen = np.zeros(len(uniq), bool)
        rep = np.zeros(len(ids), bool)
        dec_rep = []
        for pos, j in enumerate(perm):
            rep[pos] = seen[ids[j]]
            seen[ids[j]] = True
        steps = [rep[k:k + 16].mean() for k in range(0, len(ids), 16)]
        sims.append(steps)
    sims = np.array(sims)
    q = [0, 9, 19, 28, 38, 47, 56]
    out["expected_repeat_share_by_epoch1_step"] = {f"step_{s + 1}": round(float(sims[:, s].mean()), 3) for s in q}
    # decision rows only: share of decision rows whose exact target was already seen earlier in the epoch
    dids = np.array([uniq[t] for t, d in zip(tg, turn) if d])
    out["decision_rows_repeating_target_within_epoch_share"] = round(1 - len(set(dids.tolist())) / len(dids), 3)
    dump(out, "sft_log.json")
    print(json.dumps({k: v for k, v in out.items() if k != "per_step"}, indent=0))


if __name__ == "__main__":
    main()
