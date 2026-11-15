"""chain2 — the unattended driver for every remaining GPU experiment of "What They Meant" (2026-11-15).

WHY: on 2026-11-14 the v1 pipeline stopped at 21:30 IST on two eval episodes and the GPU then sat idle for 12.5 h, because the
plan depended on a Claude session being available to react. chain2 never waits for a person: it runs the step scripts in
<WS>/training/chain2_steps/*.sh one at a time in lexical order, re-scanning the directory before every step, so new steps can be
deployed while it runs (a step file is read by bash only when its step starts; never edit a step file while that step runs).
When no step is left it reports state "idle" and keeps polling for new step files; the session's watcher treats "idle" (the GPU
has nothing to do), a new attention entry and "finished" as the only reasons to wake the session.

Step protocol (each step: bash <step>.sh, cwd <WS>, its own process group, output -> <WS>/logs/chain2/<step>.log):
  exit 0  done      -> marker runs/chain2/markers/<step>.done (never rerun; delete the marker to rerun)
  exit 3  skipped   -> marker <step>.skipped with the reason (its last 'CHAIN2_SKIP <reason>' line); delete it to retry
  exit 4  not yet   -> nothing written; the driver sleeps CHAIN2_POLL_S and runs the same step again (waits on a condition)
  other   failed    -> marker <step>.failed + an attention entry; the driver continues with the next step
  A line 'CHAIN2_ATTENTION <text>' anywhere in a step's output adds an attention entry (e.g. GRPO killed but checkpoints exist).
Status: <WS>/logs/chain2.status (JSON: state running|waiting|idle|finished|aborted, step, ts, ts_ist, pid, attention [...],
steps {name: {state, rc, start_ist, end_ist, minutes, reason}}); log <WS>/logs/chain2.log; finish: touch runs/chain2/STOP
(state finished once idle). SIGTERM/SIGINT: forwarded to the running step's process group (its memguard kills its own job
tree), state aborted. Single instance: flock on runs/chain2/.lock. Never wrap this driver in memguard (each GPU step is).

  dbg chain2 "cd /workspace && python3 training/run_chain2.py"        (inside lx-exp; WTM_WS overrides /workspace for tests)
"""
from __future__ import annotations

import fcntl
import glob
import json
import os
import signal
import subprocess
import sys
import time

WS = os.environ.get("WTM_WS", "/workspace")
STEPS_DIR = os.environ.get("CHAIN2_STEPS_DIR", os.path.join(WS, "training", "chain2_steps"))
CDIR = os.path.join(WS, "runs", "chain2")
MARK = os.path.join(CDIR, "markers")
LOGDIR = os.path.join(WS, "logs", "chain2")
LOG = os.path.join(WS, "logs", "chain2.log")
STATUS = os.path.join(WS, "logs", "chain2.status")
POLL_S = float(os.environ.get("CHAIN2_POLL_S", "60"))
IDLE_POLL_S = float(os.environ.get("CHAIN2_IDLE_POLL_S", "120"))
MAX_IDLE_S = float(os.environ.get("CHAIN2_MAX_IDLE_S", str(36 * 3600)))   # give up polling after this long idle

state = {"state": "starting", "step": None, "pid": os.getpid(), "attention": [], "steps": {}, "log": LOG,
         "steps_dir": STEPS_DIR, "started_ist": None}
child: subprocess.Popen | None = None


def ist(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M IST", time.gmtime((t or time.time()) + 5.5 * 3600))


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} ({ist()}) {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def save_status(**kw) -> None:
    state.update(kw)
    state["ts"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    state["ts_ist"] = ist()
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, STATUS)


def attention(step: str, reason: str) -> None:
    state["attention"].append({"ts_ist": ist(), "step": step, "reason": reason[:500]})
    log(f"ATTENTION {step}: {reason[:300]}")
    save_status()


def marker(step: str) -> str | None:
    for kind in ("done", "skipped", "failed"):
        if os.path.exists(os.path.join(MARK, f"{step}.{kind}")):
            return kind
    return None


def write_marker(step: str, kind: str, info: dict) -> None:
    p = os.path.join(MARK, f"{step}.{kind}")
    with open(p + ".tmp", "w") as fh:
        json.dump({"step": step, "kind": kind, "ts_ist": ist(), **info}, fh, indent=1)
    os.replace(p + ".tmp", p)


def steps_on_disk() -> list[str]:
    return sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(STEPS_DIR, "*.sh")))


def scan_log(path: str, prefix: str) -> list[str]:
    out = []
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                i = line.find(prefix)
                if i >= 0:
                    out.append(line[i + len(prefix):].strip())
    except OSError:
        pass
    return out


def run_step(step: str) -> int:
    global child
    path = os.path.join(STEPS_DIR, step + ".sh")
    lp = os.path.join(LOGDIR, step + ".log")
    env = dict(os.environ, CHAIN2_STEP=step, CHAIN2_DIR=CDIR, WS=WS, PURE_TRL_MODE="1")
    with open(lp, "a") as fh:
        fh.write(f"\n##### {ist()} START {step}\n")
        fh.flush()
        child = subprocess.Popen(["bash", path], cwd=WS, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        rc = child.wait()
        child = None
        fh.write(f"##### {ist()} END {step} rc={rc}\n")
    return rc


def on_signal(signum, _frame) -> None:
    name = signal.Signals(signum).name
    log(f"CHAIN2_ABORT received {name}; forwarding to the running step")
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)   # the step's memguard kills its own job tree
            child.wait(timeout=600)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    save_status(state="aborted")
    sys.exit(143)


def main() -> None:
    for d in (CDIR, MARK, LOGDIR, os.path.dirname(STATUS)):
        os.makedirs(d, exist_ok=True)
    lockf = open(os.path.join(CDIR, ".lock"), "a+")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("CHAIN2_REFUSE another run_chain2.py holds runs/chain2/.lock", flush=True)
        sys.exit(75)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if os.path.exists(STATUS):   # keep earlier attention entries and step records across restarts
        try:
            old = json.load(open(STATUS))
            state["attention"] = old.get("attention", [])
            state["steps"] = old.get("steps", {})
        except (OSError, ValueError):
            pass
    state["started_ist"] = ist()
    log(f"CHAIN2_START pid={os.getpid()} steps_dir={STEPS_DIR} steps={steps_on_disk()}")
    save_status(state="running")
    idle_since = None
    while True:
        pending = [s for s in steps_on_disk() if marker(s) is None]
        if not pending:
            if os.path.exists(os.path.join(CDIR, "STOP")):
                log("CHAIN2_DONE (STOP file present and no step pending)")
                save_status(state="finished", step=None)
                return
            if idle_since is None:
                idle_since = time.time()
                log("CHAIN2_IDLE no step pending; polling for new step files (touch runs/chain2/STOP to finish)")
                save_status(state="idle", step=None)
            if time.time() - idle_since > MAX_IDLE_S:
                log("CHAIN2_DONE (idle limit reached)")
                save_status(state="finished", step=None)
                return
            time.sleep(IDLE_POLL_S)
            continue
        idle_since = None
        step = pending[0]
        t0 = time.time()
        rec = state["steps"].setdefault(step, {})
        rec.update({"state": "running", "start_ist": ist(t0), "end_ist": None, "rc": None})
        save_status(state="running", step=step)
        log(f"STEP_START {step}")
        rc = run_step(step)
        lp = os.path.join(LOGDIR, step + ".log")
        mins = round((time.time() - t0) / 60, 1)
        for a in scan_log(lp, "CHAIN2_ATTENTION "):
            if not any(x["step"] == step and x["reason"] == a[:500] for x in state["attention"]):
                attention(step, a)
        if rc == 4:
            rec.update({"state": "waiting", "rc": rc})
            save_status(state="waiting", step=step)
            time.sleep(POLL_S)
            continue
        if rc == 0:
            write_marker(step, "done", {"minutes": mins})
            rec.update({"state": "done", "rc": 0, "end_ist": ist(), "minutes": mins})
            log(f"STEP_DONE {step} in {mins} min")
        elif rc == 3:
            why = (scan_log(lp, "CHAIN2_SKIP ") or ["(no reason given)"])[-1]
            write_marker(step, "skipped", {"reason": why, "minutes": mins})
            rec.update({"state": "skipped", "rc": 3, "end_ist": ist(), "minutes": mins, "reason": why})
            log(f"STEP_SKIP {step}: {why}")
        else:
            write_marker(step, "failed", {"rc": rc, "minutes": mins})
            rec.update({"state": "failed", "rc": rc, "end_ist": ist(), "minutes": mins})
            log(f"STEP_FAIL {step} rc={rc} (log {lp})")
            attention(step, f"step failed rc={rc}; log {lp}")
        save_status(state="running", step=None)


if __name__ == "__main__":
    main()
