"""Print flattened key paths of a JSON artifact (helper for locating cited numbers). Usage: python3 flat.py FILE [substr ...]"""
import json, sys

def flat(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{pre}/{k}")
    elif isinstance(x, list) and x and all(isinstance(i, (dict, list)) for i in x) and len(x) <= 12:
        for i, v in enumerate(x):
            yield from flat(v, f"{pre}[{i}]")
    else:
        yield pre, x

p = sys.argv[1]
subs = [s.lower() for s in sys.argv[2:]]
for k, v in flat(json.load(open(p))):
    if subs and not any(s in k.lower() for s in subs):
        continue
    s = json.dumps(v)
    print(f"{k} = {s[:160]}")
