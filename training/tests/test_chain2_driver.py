"""Offline test of training/run_chain2.py (plain python3, temp workspace, fake steps, no GPU, ~10 s).

Covers: done / skipped / not-yet-then-done / failed (+attention) / a CHAIN2_ATTENTION line / a step deployed while the chain is
idle is picked up / STOP -> finished / markers + status JSON / a second instance refuses (75). Prints TEST_CHAIN2_DRIVER_OK.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DRIVER = os.path.join(ROOT, "training", "run_chain2.py")


def main() -> None:
    with tempfile.TemporaryDirectory() as ws:
        steps = os.path.join(ws, "training", "chain2_steps")
        os.makedirs(steps)
        w = lambda name, body: open(os.path.join(steps, name), "w").write(body)   # noqa: E731
        w("00_ok.sh", "echo hello; exit 0\n")
        w("10_skip.sh", "echo 'CHAIN2_SKIP no approval file'; exit 3\n")
        w("20_wait.sh", 'f="$WS/waited"; if [ -f "$f" ]; then exit 0; fi; touch "$f"; echo not yet; exit 4\n')
        w("30_fail.sh", "echo boom; exit 7\n")
        w("40_attn.sh", "echo 'CHAIN2_ATTENTION GRPO ended early; using checkpoint-30'; exit 0\n")
        env = dict(os.environ, WTM_WS=ws, CHAIN2_POLL_S="0.2", CHAIN2_IDLE_POLL_S="0.2", CHAIN2_MAX_IDLE_S="60")
        p = subprocess.Popen([sys.executable, DRIVER], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        status = os.path.join(ws, "logs", "chain2.status")

        def st():
            try:
                return json.load(open(status))
            except (OSError, ValueError):
                return {}

        t0 = time.time()
        while st().get("state") != "idle" and time.time() - t0 < 30:
            time.sleep(0.1)
        s = st()
        assert s.get("state") == "idle", s
        q = subprocess.run([sys.executable, DRIVER], env=env, capture_output=True, text=True)   # second instance
        assert q.returncode == 75 and "CHAIN2_REFUSE" in q.stdout, (q.returncode, q.stdout)
        w("50_late.sh", "echo deployed while idle; exit 0\n")   # picked up by the idle poll
        t0 = time.time()
        while not os.path.exists(os.path.join(ws, "runs", "chain2", "markers", "50_late.done")) and time.time() - t0 < 30:
            time.sleep(0.1)
        open(os.path.join(ws, "runs", "chain2", "STOP"), "w").close()
        out, _ = p.communicate(timeout=30)
        assert p.returncode == 0, out
        s = st()
        mk = sorted(os.listdir(os.path.join(ws, "runs", "chain2", "markers")))
        assert mk == ["00_ok.done", "10_skip.skipped", "20_wait.done", "30_fail.failed", "40_attn.done", "50_late.done"], mk
        assert s["state"] == "finished", s["state"]
        reasons = [a["reason"] for a in s["attention"]]
        assert any("rc=7" in r for r in reasons) and any("checkpoint-30" in r for r in reasons), reasons
        assert s["steps"]["10_skip"]["reason"] == "no approval file", s["steps"]["10_skip"]
        assert "CHAIN2_DONE" in out and "STEP_FAIL 30_fail rc=7" in out, out
        print(f"driver: markers {mk}; attention {len(reasons)}; final state {s['state']}")
    print("TEST_CHAIN2_DRIVER_OK")


if __name__ == "__main__":
    main()
