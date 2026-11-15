"""SEA-2 verification, refined: what does the L2s block ADD to the L2 first prompt?

Per held-out row (P0):
  new_info      a decisive_facts line has an informative token (normalised numbers, identifiers, words >= 5 letters) that is
                absent from the L2 first prompt (system + request)  -> L2s is not salience-only for that row
  lookup_info   ... and that token occurs in the canonical look-up results the eval serves (raw fields)  -> look-up outcome
  seed_copy     a fact line occurs verbatim (normalised) inside a SEED tool-rule result of a look-up/verify tool (the seed
                result text can carry verdict words that the raw results served at L0r..L2 deliberately drop)
  verdict_words the L2s block contains a verdict-like phrase absent from the raw look-up results and the L2 first prompt
Pair-level: twins whose L2 first prompts are byte-identical but whose L2s prompts differ (the block alone separates them).
Output: results/v02b_l2s_newinfo.json
"""
import json
import re
from collections import Counter

from vcommon import dump, env, episode_view, eval_sets

STOP = set("""about above after again against because before being below between could during every their there these those
through under until where which while would should shall cannot member members result results found record records""".split())
VERDICT = re.compile(r"\b(matches|matching|never paid|no photo|new account|already paid|not the same|different|duplicate|"
                     r"no subscription|no new recurring|includes new recurring|until cancell?ed|first result|scam|unverified|"
                     r"verified|not verified|no mutual|mismatch)\b", re.I)


def canon_num(s):
    try:
        f = float(s)
    except ValueError:
        return s
    t = f"{f:.2f}".rstrip("0").rstrip(".")
    return t


def toks(text):
    t = (text or "").lower().replace("’", "'").replace("_", " ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    t = re.sub(r"[$€£₹•*]", " ", t)
    out = set()
    for w in re.findall(r"[a-z0-9@][a-z0-9@.\-]*[a-z0-9]|[a-z0-9]", t):
        w = w.strip(".-")
        if re.fullmatch(r"\d+(\.\d+)?", w):
            out.add(canon_num(w))
        else:
            out.add(w)
    return out


def informative(ts):
    keep = set()
    for w in ts:
        if re.fullmatch(r"\d+(\.\d+)?", w):
            if len(w.replace(".", "")) >= 2:
                keep.add(w)
        elif re.search(r"\d", w) or "@" in w:
            keep.add(w)
        elif len(w) >= 5 and w not in STOP:
            keep.add(w)
    return keep


def normtxt(t):
    return re.sub(r"\s+", " ", (t or "").lower().replace("’", "'")).strip()


ev = eval_sets()
out, rows_out = {}, {}
for name in ("test", "hard_test", "val", "hard_val", "hard_train"):
    c = Counter()
    per = []
    pairs = {}
    for s in ev[name]:
        if s["twin"].startswith("A_pushback"):
            continue
        e = env(s["env_id"])
        kinds = {t["name"]: t["kind"] for t in e["tools"]}
        sys2, user, tools, msgs = episode_view(s, "L2", "P0")
        sys2s, _, _, _ = episode_view(s, "L2s", "P0")
        block = sys2s[len(sys2.rstrip()):]
        ctx = toks(sys2 + "\n" + user)
        res_raw = "\n".join(m["content"] for m in msgs[2::2])
        rtok = toks(res_raw)
        seed_txt = normtxt(" ".join(json.dumps(r.get("result"), ensure_ascii=False) if not isinstance(r.get("result"), str)
                                    else r["result"] for r in s.get("tool_rules", []) if kinds.get(r["tool"]) in ("lookup", "verify")))
        new_info = lookup_info = seed_copy = False
        ex = []
        for f in s.get("decisive_facts") or []:
            ft = informative(toks(f))
            new = sorted(ft - ctx)
            look = sorted((ft - ctx) & rtok)
            new_info |= bool(new)
            lookup_info |= bool(look)
            sc = len(normtxt(f)) >= 12 and normtxt(f) in seed_txt
            seed_copy |= sc
            if (new or sc) and len(ex) < 3:
                ex.append({"fact": f[:160], "new_tokens": new[:6], "lookup_tokens": look[:6], "seed_copy": sc})
        vb = {m.group(0).lower() for m in VERDICT.finditer(block)}
        vr = {m.group(0).lower() for m in VERDICT.finditer(res_raw + "\n" + sys2 + "\n" + user)}
        verdict_only_in_block = sorted(vb - vr)
        c["rows"] += 1
        c["new_info"] += new_info
        c["lookup_info"] += lookup_info
        c["seed_copy"] += seed_copy
        c["new_or_seed_copy"] += (new_info or seed_copy)
        c["verdict_words_only_in_L2s_block"] += bool(verdict_only_in_block)
        per.append({"id": s["id"], "new_info": new_info, "lookup_info": lookup_info, "seed_copy": seed_copy,
                    "verdict_only_in_block": verdict_only_in_block, "examples": ex})
        pairs.setdefault(s["pair_id"], {})[s["twin"][0]] = (sys2 + "\x00" + user, sys2s + "\x00" + user)
    same_l2 = [p for p, t in pairs.items() if len(t) == 2 and t["A"][0] == t["B"][0]]
    same_l2s = [p for p, t in pairs.items() if len(t) == 2 and t["A"][1] == t["B"][1]]
    c["pairs"] = sum(len(t) == 2 for t in pairs.values())
    c["pairs_identical_L2_first_prompt"] = len(same_l2)
    c["pairs_identical_L2s_first_prompt"] = len(same_l2s)
    c["rows_in_pairs_separated_only_by_L2s_block"] = 2 * len(set(same_l2) - set(same_l2s))
    out[name] = dict(c) | {"pairs_identical_at_L2": same_l2}
    rows_out[name] = per

dump("v02b_l2s_newinfo.json", {"counts": out, "rows": rows_out})
for k, v in out.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "pairs_identical_at_L2"})
