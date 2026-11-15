#!/usr/bin/env python3
"""memguard: run ONE GPU job inside lx-exp in its own session/process group and SIGKILL that job's process tree before it can
starve the shared DGX Spark. Why: GB10 has 121 GB of unified memory, so CUDA allocations come out of host RAM; an overrun does
not raise a CUDA OOM, it pushes the whole box (other people's production containers, sshd) into reclaim and thrash. That is how
the 2026-11-14 smoke run hung the host.

Safety scope: memguard only ever signals (a) the process group it created for its child (start_new_session=True, so
pgid == child pid; never 0, 1 or memguard's own group) and (b) processes whose parent chain leads back to memguard itself.
On Linux memguard makes itself a child subreaper (PR_SET_CHILD_SUBREAPER), so a grandchild that calls setsid() and is orphaned
reparents to memguard and stays inside (b). It never matches processes by name (no pkill), so it cannot touch other containers
even if lx-exp runs with --pid=host.

Kill conditions, checked every --poll seconds:
  * baseline MemAvailable - MemAvailable > --budget-gb       (this job's budget; other tenants' growth also counts: safe side)
  * host MemAvailable < --floor-gb                           (box-wide floor; with the preflight below the budget normally fires
                                                              first, the floor names the reason when the box itself is low)
  * optional: /proc/pressure/memory "full avg10" > --psi-full (the box is stalling on memory)
  * optional: major faults/s > --thrash-majflt for 3 s in a row (fallback where PSI is missing)
  * wall time > --max-minutes: SIGTERM, then SIGKILL after --grace-s (memory stays guarded during the grace period)
Preflight: exit 75 if another memguard job holds --lock, or MemAvailable < budget + floor.
Child setup: own session, oom_score_adj=1000, PR_SET_PDEATHSIG=SIGKILL (the job dies if memguard dies), WTM_GPU_CAP_GB exported
from --gpu-cap-gb (training/guards.py apply_mem_cap turns it into a torch allocator cap).
After the child exits (or is killed) memguard SIGKILLs any process left in its tree (e.g. an orphaned vLLM EngineCore) and waits
until the tree is gone, so the next stage never starts while this one's memory is still held.
Exit: the child's rc (128+N if killed by signal N); 137 after a memguard kill (stderr "MEMGUARD_KILL ..."); 75 on refusal.
An error after the child has exited (summary/status write, final meminfo) is logged as MEMGUARD_TAIL_ERROR and never changes it.
Status file <status-dir>/memguard-<name>.status (JSON, rewritten every 30 s and at the end: one remote `cat`, no polling loop over
ssh). Log <log>. Summary JSON via --summary.

Usage (inside lx-exp):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 training/memguard.py --name m3_dpo --budget-gb 70 --gpu-cap-gb 60 \
      --max-minutes 720 --summary /workspace/runs/m3_dpo/memguard.json -- python3 training/train_dpo.py ...
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import resource
import signal
import subprocess
import sys
import time

GB = 1024 ** 3
PR_SET_PDEATHSIG, PR_SET_CHILD_SUBREAPER = 1, 36
MCL_CURRENT, MCL_FUTURE = 1, 2
REFUSED, KILLED = 75, 137


def parse_args():
    # Split at the first "--" ourselves: argparse's REMAINDER/"--" handling differs between Python versions, and the child's
    # own arguments (which may contain "--") must reach it untouched.
    argv = sys.argv[1:]
    ap = argparse.ArgumentParser(usage="memguard.py [options] -- cmd ...")
    ap.add_argument("--name", required=True)
    ap.add_argument("--budget-gb", type=float, required=True)
    ap.add_argument("--floor-gb", type=float, default=16.0)
    ap.add_argument("--gpu-cap-gb", type=float, default=0, help="exported to the child as WTM_GPU_CAP_GB (0 = not set)")
    ap.add_argument("--max-minutes", type=float, default=0, help="0 = no wall-time limit")
    ap.add_argument("--grace-s", type=float, default=30, help="SIGTERM -> SIGKILL grace for the wall-time limit")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--psi-full", type=float, default=0, help="kill if memory PSI full avg10 exceeds this %% (0 = off)")
    ap.add_argument("--thrash-majflt", type=float, default=0, help="major faults/s for 3 s in a row that count as thrashing")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--lock", default="/workspace/.gpu_job.lock")
    ap.add_argument("--log", default="/workspace/logs/memguard.log")
    ap.add_argument("--status-dir", default="/workspace/logs")
    ap.add_argument("--meminfo", default="/proc/meminfo", help=argparse.SUPPRESS)   # offline tests feed a synthetic file
    if "--" not in argv:
        ap.error("no command given")
    i = argv.index("--")
    ns = ap.parse_args(argv[:i])
    ns.cmd = argv[i + 1:]
    if not ns.cmd:
        ap.error("no command given after --")
    return ns


ns = parse_args()
os.makedirs(os.path.dirname(os.path.abspath(ns.log)), exist_ok=True)
os.makedirs(ns.status_dir, exist_ok=True)
logf = open(ns.log, "a", buffering=1)
MYPID, MYPGID = os.getpid(), os.getpgid(0)


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{ns.name}] {msg}"
    for fh in (logf, sys.stderr):   # a full disk or a closed stderr must not turn a log line into a crash
        if fh is None:              # started with fd 2 closed: sys.stderr is None
            continue
        try:
            fh.write(line + "\n")
            fh.flush()
        except (OSError, ValueError, AttributeError):
            pass


try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    _libc = None
    if sys.platform.startswith("linux"):
        log("WARNING libc.so.6 not loadable: no subreaper, no PDEATHSIG, no mlock")

if _libc:
    # Keep the watchdog resident. MCL_FUTURE only when RLIMIT_MEMLOCK is unlimited: under a finite limit a later allocation in
    # the watchdog itself would fail (MemoryError) exactly when memory is tight.
    if resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] == resource.RLIM_INFINITY:
        _libc.mlockall(MCL_CURRENT | MCL_FUTURE)   # best effort
    if _libc.prctl(PR_SET_CHILD_SUBREAPER, ctypes.c_ulong(1), 0, 0, 0) != 0:
        log("WARNING PR_SET_CHILD_SUBREAPER failed; setsid-escaped grandchildren may be missed")

_mi_fd = os.open(ns.meminfo, os.O_RDONLY)   # opened once; non-inheritable (PEP 446)


def meminfo():
    # A few retries only matter for the offline tests' synthetic file, which is rewritten in place while we read it.
    for _ in range(20):
        out = {}
        for ln in os.pread(_mi_fd, 16384, 0).decode(errors="replace").splitlines():
            k, _, v = ln.partition(":")
            parts = v.split()
            if parts and parts[0].isdigit():
                out[k.strip()] = int(parts[0]) * 1024
        if "MemAvailable" in out and "MemTotal" in out:
            return out
        time.sleep(0.005)
    raise RuntimeError(f"{ns.meminfo} has no MemTotal/MemAvailable")


def psi_full():
    try:
        with open("/proc/pressure/memory") as fh:
            for ln in fh:
                if ln.startswith("full "):
                    return float(ln.split("avg10=")[1].split()[0])
    except (OSError, IndexError, ValueError):
        pass
    return None


def majflt():
    try:
        with open("/proc/vmstat") as fh:
            for ln in fh:
                if ln.startswith("pgmajfault "):
                    return int(ln.split()[1])
    except (OSError, ValueError):
        pass
    return None


def descendants(root):
    """[(pid, state)] of every process whose parent chain reaches root; empty where /proc is missing (macOS)."""
    kids = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return []
    for d in names:
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as fh:
                st = fh.read()
            state, ppid = st[st.rfind(")") + 2:].split()[:2]
            kids.setdefault(int(ppid), []).append((int(d), state))
        except (OSError, ValueError, IndexError):
            continue
    out, stack = [], [root]
    while stack:
        for pid, state in kids.get(stack.pop(), []):
            out.append((pid, state))
            stack.append(pid)
    return out


# ---- preflight: one guarded GPU job at a time inside lx-exp, and enough memory for budget + floor ----
os.makedirs(os.path.dirname(os.path.abspath(ns.lock)), exist_ok=True)
lock_fh = open(ns.lock, "a+")
try:
    fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    lock_fh.seek(0)
    log(f"MEMGUARD_REFUSE another guarded GPU job holds {ns.lock}: {lock_fh.read().strip()!r}")
    sys.exit(REFUSED)
lock_fh.seek(0)
lock_fh.truncate()
lock_fh.write(f"{ns.name} pid={MYPID} since={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
lock_fh.flush()

m0 = meminfo()
base = m0["MemAvailable"]
psi0 = psi_full()
log(f"preflight MemTotal={m0['MemTotal'] / GB:.1f}G MemAvailable={base / GB:.1f}G MemFree={m0.get('MemFree', 0) / GB:.1f}G "
    f"SwapTotal={m0.get('SwapTotal', 0) / GB:.1f}G psi_full={psi0} budget={ns.budget_gb}G floor={ns.floor_gb}G "
    f"cap={ns.gpu_cap_gb or os.environ.get('WTM_GPU_CAP_GB') or '-'}G cmd={' '.join(ns.cmd)[:300]}")
if ns.meminfo == "/proc/meminfo" and m0["MemTotal"] < 100 * GB:
    log(f"WARNING MemTotal {m0['MemTotal'] / GB:.1f}G: /proc/meminfo looks virtualised (LXCFS?); the floor is not host-wide")
if ns.psi_full and psi0 is None:
    log("WARNING /proc/pressure/memory unavailable: --psi-full is inactive")
if base < (ns.budget_gb + ns.floor_gb) * GB:
    log(f"MEMGUARD_REFUSE MemAvailable {base / GB:.1f}G < budget {ns.budget_gb}G + floor {ns.floor_gb}G; not starting")
    sys.exit(REFUSED)

env = dict(os.environ)
if ns.gpu_cap_gb:
    env["WTM_GPU_CAP_GB"] = str(ns.gpu_cap_gb)


def _child_setup():   # runs in the child between fork and exec
    if _libc:
        _libc.prctl(PR_SET_PDEATHSIG, ctypes.c_ulong(int(signal.SIGKILL)), 0, 0, 0)
        if os.getppid() != MYPID:   # memguard died before prctl took effect
            os._exit(1)
    try:
        with open("/proc/self/oom_score_adj", "w") as fh:
            fh.write("1000")   # the kernel OOM killer picks our job first (raising it needs no privilege)
    except OSError:
        pass


t0 = time.time()
child = subprocess.Popen(ns.cmd, env=env, start_new_session=True, preexec_fn=_child_setup)
pgid = child.pid
if pgid in (0, 1, MYPID, MYPGID):
    log(f"refusing to manage pgid {pgid}")
    child.kill()
    sys.exit(70)
log(f"started pid/pgid={pgid}")
state = {"reasons": [], "killed": False, "rc": None, "min_avail": base, "max_used": 0, "max_psi": 0.0}


def signal_tree(sig):
    # The group first: on a starving host it must not wait for a full /proc walk. Processes it kills that had children leave
    # orphans that the subreaper keeps under memguard, so the walk afterwards still finds everything outside the group.
    try:
        os.killpg(pgid, sig)                       # our job's own group only
    except (ProcessLookupError, PermissionError):
        pass
    victims = [p for p, _ in descendants(MYPID)]   # everything under memguard (the subreaper keeps orphans here)
    for p in victims:
        if p in (0, 1, MYPID):
            continue
        try:
            os.kill(p, sig)
        except (ProcessLookupError, PermissionError):
            pass
    return len(victims)


def kill_now(reason):
    state["reasons"].append(reason)
    state["killed"] = True
    n = signal_tree(signal.SIGKILL)
    log(f"MEMGUARD_KILL reason={reason} pgid={pgid} tree_pids={n}")
    print(f"MEMGUARD_KILL {ns.name}: {reason}", file=sys.stderr, flush=True)


def on_signal(signum, _frame):
    if not state["killed"]:
        kill_now(f"memguard received signal {signum}")


for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    # nohup'd jobs (connect.sh dbg) ignore SIGHUP; keep it ignored so a dropped session never kills a long run
    if _s == signal.SIGHUP and signal.getsignal(_s) == signal.SIG_IGN:
        continue
    signal.signal(_s, on_signal)


def write_status(d):
    p = os.path.join(ns.status_dir, f"memguard-{ns.name}.status")
    try:
        with open(p + ".tmp", "w") as fh:
            json.dump(d, fh)
        os.replace(p + ".tmp", p)
    except OSError:
        pass


def tree_alive():
    """Live (non-zombie) processes still in our tree or our job's group."""
    left = [p for p, st in descendants(MYPID) if st != "Z"]
    try:
        os.killpg(pgid, 0)
        group = True
    except (ProcessLookupError, PermissionError):
        group = False
    return left, group


def reap():
    """Reap zombies reparented to us (subreaper); only called once Popen has finished with its own child."""
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
        if pid == child.pid and state["rc"] is None:
            state["rc"] = os.waitstatus_to_exitcode(status)


last_beat = 0.0
term_deadline = None
mf_prev, mf_t, hot = majflt(), time.time(), 0
try:
    while True:
        rc = child.poll()
        if rc is not None:
            state["rc"] = rc
            break
        m = meminfo()
        avail = m["MemAvailable"]
        used = base - avail
        now = time.time()
        state["min_avail"] = min(state["min_avail"], avail)
        state["max_used"] = max(state["max_used"], used)
        if not state["killed"]:
            if avail < ns.floor_gb * GB:
                kill_now(f"host MemAvailable {avail / GB:.1f}G < floor {ns.floor_gb}G (drop since start {used / GB:.1f}G)")
            elif used > ns.budget_gb * GB:
                kill_now(f"MemAvailable drop {used / GB:.1f}G > budget {ns.budget_gb}G (MemAvailable {avail / GB:.1f}G)")
            elif term_deadline is not None and now > term_deadline:
                kill_now(f"still running {ns.grace_s:.0f}s after SIGTERM")
            elif ns.max_minutes and term_deadline is None and now - t0 > ns.max_minutes * 60:
                state["reasons"].append(f"wall time {(now - t0) / 60:.1f} min > {ns.max_minutes} min")
                n = signal_tree(signal.SIGTERM)
                term_deadline = now + ns.grace_s
                log(f"MEMGUARD_TERM wall time {(now - t0) / 60:.1f} min > {ns.max_minutes} min: SIGTERM to {n} pids + group "
                    f"{pgid}, SIGKILL in {ns.grace_s:.0f}s")
            elif now - mf_t >= 1.0:
                pf = psi_full()
                if pf is not None:
                    state["max_psi"] = max(state["max_psi"], pf)
                    if ns.psi_full and pf > ns.psi_full:
                        kill_now(f"memory PSI full avg10 {pf:.1f}% > {ns.psi_full}% (MemAvailable {avail / GB:.1f}G)")
                if not state["killed"] and ns.thrash_majflt:
                    mf = majflt()
                    if mf is not None and mf_prev is not None:
                        rate = (mf - mf_prev) / (now - mf_t)
                        hot = hot + 1 if rate > ns.thrash_majflt else 0
                        if hot >= 3:
                            kill_now(f"thrashing: {rate:.0f} major faults/s for 3 s (MemAvailable {avail / GB:.1f}G)")
                    mf_prev = mf
                mf_t = now
        if state["killed"]:
            try:
                state["rc"] = child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                log("child not reaped 60 s after SIGKILL (D state in the GPU driver?)")
            break
        if now - last_beat >= 30:
            beat = {"t_min": round((now - t0) / 60, 1), "avail_gb": round(avail / GB, 1), "used_gb": round(used / GB, 1),
                    "peak_used_gb": round(state["max_used"] / GB, 1), "min_avail_gb": round(state["min_avail"] / GB, 1),
                    "swapfree_gb": round(m.get("SwapFree", 0) / GB, 1), "psi_full_max": state["max_psi"], "pgid": pgid}
            log("beat " + json.dumps(beat))
            write_status(beat | {"name": ns.name, "state": "running", "ts": time.strftime("%H:%M:%S")})
            last_beat = now
        time.sleep(ns.poll)
except Exception as exc:   # never leave the job running unguarded
    if not state["killed"]:
        kill_now(f"memguard internal error {exc!r}")

# Whatever is left in our tree after the child exits (e.g. an orphaned vLLM EngineCore) is ours: kill it and wait until it is
# gone, so its memory is back before the caller starts the next stage. The tail is wrapped so that a failure in it (a starved
# /proc read, a full disk, a bad inherited WTM_GPU_CAP_GB) is logged and memguard still exits with the job's rc: the caller's
# stop-on-failure logic reads that rc, not memguard's own errors.
tree_gone = None
try:
    left, group = tree_alive()
    if left or group:
        log(f"stragglers after exit: group_alive={group} pids={left[:20]} -> SIGKILL")
        signal_tree(signal.SIGKILL)
    deadline = time.time() + 60
    while True:
        reap()
        left, group = tree_alive()
        if not (left or group) or time.time() > deadline:
            break
        time.sleep(0.2)
    tree_gone = not (left or group)
    if not tree_gone:
        log(f"MEMGUARD_LEFTOVER processes survive SIGKILL after 60 s (D state?): group_alive={group} pids={left[:20]}; "
            "check before starting another GPU job")
except Exception as exc:
    try:   # the cleanup itself failed: still kill our own group, never leave it holding memory
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass
    tree_gone = False
    log(f"MEMGUARD_TAIL_ERROR straggler cleanup: {exc!r}")
    log("MEMGUARD_LEFTOVER cleanup failed; tree state unknown — check before starting another GPU job")

rc = state["rc"]
exit_rc = KILLED if state["reasons"] else (1 if rc is None else (128 - rc if rc < 0 else rc))
try:
    time.sleep(3)   # unified memory is returned asynchronously after process teardown
    after = meminfo()["MemAvailable"]
    cap = env.get("WTM_GPU_CAP_GB")
    try:
        cap = float(cap) if cap else None
    except ValueError:
        pass   # recorded as given (guards.apply_mem_cap fails a GPU script on it)
    summary = {"name": ns.name, "cmd": ns.cmd, "rc": state["rc"], "exit_rc": exit_rc,
               "kill_reason": "; ".join(state["reasons"]) or None,
               "minutes": round((time.time() - t0) / 60, 2), "baseline_avail_gb": round(base / GB, 2),
               "peak_used_gb": round(state["max_used"] / GB, 2), "min_avail_gb": round(state["min_avail"] / GB, 2),
               "avail_after_exit_gb": round(after / GB, 2), "not_recovered_gb": round((base - after) / GB, 2),
               "max_psi_full_avg10": state["max_psi"] if psi0 is not None else None, "budget_gb": ns.budget_gb,
               "floor_gb": ns.floor_gb, "gpu_cap_gb": cap, "tree_gone": tree_gone}
    log("summary " + json.dumps(summary))
    write_status(summary | {"state": "finished", "ts": time.strftime("%H:%M:%S")})
    if ns.summary:
        os.makedirs(os.path.dirname(os.path.abspath(ns.summary)), exist_ok=True)
        with open(ns.summary + ".tmp", "w") as fh:
            json.dump(summary, fh, indent=1)
        os.replace(ns.summary + ".tmp", ns.summary)
    if summary["not_recovered_gb"] > 4:
        log(f"WARNING {summary['not_recovered_gb']}G not returned after exit; check for leaked processes before the next stage")
except Exception as exc:
    log(f"MEMGUARD_TAIL_ERROR summary: {exc!r}; exiting with the job's rc {exit_rc}")
try:
    fcntl.flock(lock_fh, fcntl.LOCK_UN)
except OSError:
    pass   # released at process exit anyway
sys.exit(exit_rc)
