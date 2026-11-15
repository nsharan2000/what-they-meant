"""Stage 0 of every smoke run: measure, with no model loaded, how a CUDA allocation shows up on the GB10 host, and whether the
torch allocator cap works. Why: training/memguard.py kills a job on the drop in /proc/meminfo MemAvailable, and
guards.apply_mem_cap() relies on torch.cuda.set_per_process_memory_fraction raising a clean torch.OutOfMemoryError. On unified
memory neither is documented; if MemAvailable does not move when CUDA memory is filled, memguard is blind, and if the cap does not
raise, an overrun drains host RAM instead of failing. Both must be known before any model stage runs.

Records (JSON to --out): /proc/meminfo MemTotal/MemAvailable/MemFree/Unevictable/Mlocked/SwapTotal/SwapFree, VmRSS, cgroup v2
memory.current/memory.max, torch.cuda.mem_get_info() before, while holding, and after freeing --gib GiB (allocated and filled on
cuda); then the cap test in this process (PYTORCH_CUDA_ALLOC_CONF as given) and in a child process with the opposite
expandable_segments setting (SFT/merge/DPO run with it, GRPO without); RLIMIT_MEMLOCK (ulimit -l) and /proc/pressure/memory.
Prints MEM_ACCOUNTING_OK with the key deltas and exits 0 once measured (exit 1 only if CUDA is unavailable). The verdict fields
memavailable_tracks_cuda and cap_raises_oom are for the caller to gate on (training/smoke_aya.sh does).

Run inside lx-exp under memguard (budget 15):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 training/memguard.py --name mem0 --budget-gb 15 --gpu-cap-gb 10 -- \
      python3 training/mem_accounting_test.py --out /workspace/runs/smoke_aya/mem_accounting.json
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GiB = 2 ** 30
MEMINFO_KEYS = ("MemTotal", "MemAvailable", "MemFree", "Unevictable", "Mlocked", "SwapTotal", "SwapFree")
EXPANDABLE = "expandable_segments:True"

ap = argparse.ArgumentParser()
ap.add_argument("--gib", type=float, default=8.0, help="GiB to allocate and fill on cuda")
ap.add_argument("--out", default=None)
ap.add_argument("--cap-only", action="store_true", help=argparse.SUPPRESS)   # child mode: only the cap test, one JSON line
ns = ap.parse_args()

import torch  # noqa: E402
import guards  # noqa: E402

cap_frac = guards.apply_mem_cap()


def read_kib_file(path, keys):
    out = {}
    try:
        with open(path) as fh:
            for ln in fh:
                k, _, v = ln.partition(":")
                if k in keys and v.split():
                    out[k] = round(int(v.split()[0]) / 2 ** 20, 3)   # kB -> GiB
    except OSError:
        pass
    return out


def read_cgroup(name):
    try:
        with open(f"/sys/fs/cgroup/{name}") as fh:
            v = fh.read().strip()
        return v if v == "max" else round(int(v) / GiB, 3)
    except (OSError, ValueError):
        return None


def snapshot():
    s = read_kib_file("/proc/meminfo", MEMINFO_KEYS)
    s |= read_kib_file("/proc/self/status", ("VmRSS",))
    s["cgroup_memory_current"] = read_cgroup("memory.current")
    s["cgroup_memory_max"] = read_cgroup("memory.max")
    free, total = torch.cuda.mem_get_info()
    s["cuda_free"], s["cuda_total"] = round(free / GiB, 3), round(total / GiB, 3)
    s["torch_reserved"] = round(torch.cuda.memory_reserved() / GiB, 3)
    return s


def cap_test(gib):
    """Cap the allocator at half the request, then ask for the full request: a working cap raises torch.OutOfMemoryError."""
    total = torch.cuda.get_device_properties(0).total_memory
    frac = 0.5 * gib * GiB / total
    torch.cuda.set_per_process_memory_fraction(frac)
    oom = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    res = {"alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""), "fraction": round(frac, 5), "request_gib": gib}
    try:
        y = torch.empty(int(gib * GiB), dtype=torch.uint8, device="cuda")
        y.fill_(1)
        torch.cuda.synchronize()
        res |= {"raised_oom": False, "error": None}
        del y
    except oom as e:
        res |= {"raised_oom": True, "error": str(e).splitlines()[0][:200]}
    except RuntimeError as e:   # some other failure: the cap did not produce the catchable OOM the trainers rely on
        res |= {"raised_oom": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    torch.cuda.empty_cache()
    return res


if not torch.cuda.is_available():
    print("MEM_ACCOUNTING_FAIL CUDA is not available", flush=True)
    sys.exit(1)

if ns.cap_only:
    print(json.dumps(cap_test(ns.gib)), flush=True)
    sys.exit(0)

torch.zeros(1, device="cuda")   # create the CUDA context first, so the deltas below are the allocation alone
torch.cuda.synchronize()
before = snapshot()
x = torch.empty(int(ns.gib * GiB), dtype=torch.uint8, device="cuda")
x.fill_(1)   # touch every page: on unified memory an untouched allocation may not be backed yet
torch.cuda.synchronize()
time.sleep(1)
holding = snapshot()
del x
torch.cuda.empty_cache()
torch.cuda.synchronize()
time.sleep(1)
freed = snapshot()

cap_here = cap_test(ns.gib)
other_conf = "" if EXPANDABLE in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") else EXPANDABLE
env = {k: v for k, v in os.environ.items() if k not in ("PYTORCH_CUDA_ALLOC_CONF", "WTM_GPU_CAP_GB")}
if other_conf:
    env["PYTORCH_CUDA_ALLOC_CONF"] = other_conf
p = subprocess.run([sys.executable, os.path.abspath(__file__), "--cap-only", "--gib", str(ns.gib)], env=env,
                   capture_output=True, text=True, timeout=600)
try:
    cap_other = json.loads(p.stdout.strip().splitlines()[-1])
except (IndexError, ValueError):
    cap_other = {"alloc_conf": other_conf, "raised_oom": None, "error": f"child rc={p.returncode}: {p.stderr[-300:]}"}


def delta(a, b, k):
    return None if a.get(k) is None or b.get(k) is None or isinstance(a[k], str) or isinstance(b[k], str) else round(a[k] - b[k], 3)


deltas = {"memavailable_drop_gib": delta(before, holding, "MemAvailable"), "memfree_drop_gib": delta(before, holding, "MemFree"),
          "vmrss_rise_gib": delta(holding, before, "VmRSS"),
          "cgroup_current_rise_gib": delta(holding, before, "cgroup_memory_current"),
          "cuda_free_drop_gib": delta(before, holding, "cuda_free"), "unevictable_rise_gib": delta(holding, before, "Unevictable"),
          "mlocked_rise_gib": delta(holding, before, "Mlocked"),
          "memavailable_not_returned_gib": delta(before, freed, "MemAvailable")}


def lim(v):   # KiB, as `ulimit -l` prints it
    return "unlimited" if v == resource.RLIM_INFINITY else v // 1024


soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
psi = None
if os.path.exists("/proc/pressure/memory"):
    with open("/proc/pressure/memory") as fh:
        psi = fh.read().strip()
drop = deltas["memavailable_drop_gib"]
verdict = {"memavailable_tracks_cuda": drop is not None and drop >= 0.75 * ns.gib,
           "cap_raises_oom": cap_here["raised_oom"] is True,
           "cap_raises_oom_other_conf": cap_other.get("raised_oom") is True}
res = {"gib": ns.gib, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
       "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""), "wtm_gpu_cap_gb": os.environ.get("WTM_GPU_CAP_GB"),
       "apply_mem_cap_fraction": cap_frac, "before": before, "holding": holding, "after_free": freed, "deltas": deltas,
       "cap_test": cap_here, "cap_test_other_conf": cap_other,
       "ulimit_l": {"soft": lim(soft), "hard": lim(hard)}, "psi_memory_exists": psi is not None, "psi_memory": psi,
       "verdict": verdict, "peak_mem": guards.peak_mem()}
if ns.out:
    os.makedirs(os.path.dirname(os.path.abspath(ns.out)), exist_ok=True)
    with open(ns.out, "w") as fh:
        json.dump(res, fh, indent=1)
print("MEM_ACCOUNTING_OK " + json.dumps({k: v for k, v in deltas.items() if v is not None} | verdict
                                         | {"swap_total_gib": before.get("SwapTotal"), "ulimit_l": lim(soft),
                                            "psi": psi is not None}), flush=True)
