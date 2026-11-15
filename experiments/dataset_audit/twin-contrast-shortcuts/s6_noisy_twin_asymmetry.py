"""Step 6: L3 renders the scenario's user_message_noisy. Twins whose CLEAN requests are identical (or differ only in identifiers)
should get noisy rewrites that differ in the same way; otherwise L3 adds a non-decisive twin difference (tone, code-mixing).
Compares the noisy rewrites of every pair (all splits) after masking identifiers, and lists the words only one twin has.
python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s6_noisy_twin_asymmetry.py -> results/s6_noisy_twin_asymmetry.json
"""
import sys
sys.dont_write_bytecode = True
import json, os, re  # noqa: E401,E402
from collections import defaultdict, Counter  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, jdump, load_specs, load_split, twin_letter  # noqa: E402
from s1_twin_diff import IDENT  # noqa: E402


def masked(t):
    return re.sub(r"\s+", " ", IDENT.sub("<ID>", t.lower())).strip()


def main():
    specs, origin = load_specs()
    split = load_split()
    pairs = defaultdict(dict)
    for s in specs.values():
        pairs[s["pair_id"]][twin_letter(s["twin"])] = s
    rows, tab = [], Counter()
    for pid, d in sorted(pairs.items()):
        if "A" not in d or "B" not in d:
            continue
        a, b = d["A"], d["B"]
        na, nb = a.get("user_message_noisy"), b.get("user_message_noisy")
        if not na or not nb:
            continue
        sp = split.get(pid) or origin[a["id"]]
        clean_same = masked(a["user_message"]) == masked(b["user_message"])
        noisy_same = masked(na) == masked(nb)
        wa, wb = set(re.findall(r"[a-z]+", masked(na))), set(re.findall(r"[a-z]+", masked(nb)))
        ca, cb = set(re.findall(r"[a-z]+", masked(a["user_message"]))), set(re.findall(r"[a-z]+", masked(b["user_message"])))
        extra_a = sorted((wa - wb) - (ca - cb))      # words only A's rewrite adds, beyond the clean-request difference
        extra_b = sorted((wb - wa) - (cb - ca))
        key = f"{sp}|clean_{'same' if clean_same else 'diff'}|noisy_{'same' if noisy_same else 'diff'}"
        tab[key] += 1
        rows.append({"pair_id": pid, "split": sp, "clean_same_after_id_mask": clean_same, "noisy_same_after_id_mask": noisy_same,
                     "noisy_A": na, "noisy_B": nb, "clean_A": a["user_message"], "clean_B": b["user_message"],
                     "extra_words_A": extra_a, "extra_words_B": extra_b})
    asym = [r for r in rows if r["clean_same_after_id_mask"] and not r["noisy_same_after_id_mask"]]
    out = {"counts": dict(sorted(tab.items())),
           "pairs_clean_same_but_noisy_differs": {sp: sum(1 for r in asym if r["split"] == sp) for sp in sorted({r["split"] for r in rows})},
           "pairs_with_noisy_by_split": dict(Counter(r["split"] for r in rows)),
           "examples": asym[:12], "rows": rows}
    jdump(out, os.path.join(RES, "s6_noisy_twin_asymmetry.json"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "examples")}, indent=1))
    for r in asym:
        print(r["split"], r["pair_id"], "| A:", r["noisy_A"], "| B:", r["noisy_B"], "| extraA", r["extra_words_A"], "extraB", r["extra_words_B"])


if __name__ == "__main__":
    main()
