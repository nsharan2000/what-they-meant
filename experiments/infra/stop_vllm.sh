#!/bin/bash
# stop_vllm.sh [name] — stop the vLLM server(s) started by serve_vllm.sh and WAIT until they are really gone (prints free -g).
# Why: a server that was signalled but has not exited yet (or its orphaned VLLM::EngineCore child) still holds its whole
# --gpu-memory-utilization pool of the shared box's unified memory; starting the next GPU job on top of it is how the box thrashes
# (log 2026-11-14 00:47: an idle server held ~59 GB for 7 h). Stopping is by PID, from the files serve_vllm.sh writes
# (/workspace/logs/vllm-<name>.pid, port in vllm-<name>.port), never by name alone: if lx-exp ever shares the host PID namespace,
# `pkill -f "vllm serve"` would kill other projects' servers. Without a name, every recorded server is stopped and, as a fallback
# only, any real vLLM process left in OUR cgroup (catches servers started by hand and orphaned engine cores); with a name, that
# fallback is limited to orphaned engine cores (their server died, so they hold memory and no PID file). "Real" means argv[0] is
# the vllm entry point (optionally run by python) followed by 'serve', or a VLLM:: process title (setproctitle: EngineCore,
# workers, a retitled API server): a shell, grep or wrapper whose command line merely contains the text is never signalled.
# SIGTERM (vLLM shuts its engine down cleanly), SIGKILL after 45 s, give up after 60 s.
# Env: VLLM_PORT = an extra port that must be closed (serve_vllm.sh passes the port it is about to bind).
# Exit 0 = no process of ours left and every port closed; 1 = VLLM_STOP_FAILED (PID files are kept so a retry finds them).
set -uo pipefail
LOGS="${VLLM_LOG_DIR:-/workspace/logs}"
NAME="${1:-}"
PAT='[v]llm serve|VLLM::'   # pgrep prefilter only; is_vllm decides
VLLM_RE='^([^ ]*[Pp]ython[0-9.]* +(-[^ ]+ +)*)?[^ ]*vllm serve( |$)|^VLLM::'

# a zombie holds no memory and stays until its parent reaps it (a container's PID 1 may never do so), so it counts as gone
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) && [ -n "$s" ] && [ "${s#Z}" = "$s" ]; }
# also guards against a stale PID file whose PID now belongs to an unrelated process
is_vllm() { local a; a=$(ps -o args= -p "$1" 2>/dev/null) && a="${a#"${a%%[! ]*}"}" && [[ "$a" =~ $VLLM_RE ]]; }
same_cgroup() { [ ! -r /proc/self/cgroup ] || [ "$(cat "/proc/$1/cgroup" 2>/dev/null)" = "$(cat /proc/self/cgroup)" ]; }
descendants() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; descendants "$c"; done; }
# curl exit 7 = connection refused, i.e. nothing listens; anything else (even a timeout) means the port is still taken
port_open() { curl -s -o /dev/null --noproxy '*' --max-time 2 "http://127.0.0.1:$1/"; [ $? -ne 7 ]; }
# an engine core whose parent is no longer a vllm server (API server crashed or was OOM-killed, core reparented) belongs to no
# PID file, so it is swept even by a named stop; other live servers' cores are left alone
orphan_core() { ps -o args= -p "$1" 2>/dev/null | grep -q '^ *VLLM::EngineCore' \
  && ! is_vllm "$(ps -o ppid= -p "$1" 2>/dev/null | tr -d ' ')"; }
strays() {
  local p
  for p in $(pgrep -f "$PAT" 2>/dev/null); do
    [ "$p" != $$ ] && alive "$p" && is_vllm "$p" && same_cgroup "$p" && { [ -z "$NAME" ] || orphan_core "$p"; } && echo "$p"
  done
}

FILES=""; ROOTS=""; PORTS="${VLLM_PORT:-}"
if [ -n "$NAME" ]; then CAND="$LOGS/vllm-$NAME.pid"; else CAND=$(ls "$LOGS"/vllm-*.pid 2>/dev/null); fi
for f in $CAND; do
  [ -f "$f" ] || continue
  FILES="$FILES $f"
  [ -f "${f%.pid}.port" ] && PORTS="$PORTS $(cat "${f%.pid}.port")"
  pid=$(cat "$f")
  if [[ "$pid" =~ ^[0-9]+$ ]] && alive "$pid" && is_vllm "$pid"; then ROOTS="$ROOTS $pid"; fi
done

collect() {   # every live process of ours: recorded servers, their children (EngineCore, workers), fallback strays
  local p
  for p in $ROOTS; do alive "$p" && echo "$p"; descendants "$p"; done
  strays
}
TARGETS=""; SENT=" "; LEFT=""; BUSY=""
for i in $(seq 1 60); do
  # re-collect each round: an engine core spawned while the server was starting must not escape
  TARGETS=$(echo $TARGETS $(collect) | tr ' ' '\n' | sort -un | tr '\n' ' ')
  LEFT=""; for p in $TARGETS; do alive "$p" && LEFT="$LEFT $p"; done
  BUSY=""; for pt in $(echo $PORTS | tr ' ' '\n' | sort -u); do port_open "$pt" && BUSY="$BUSY $pt"; done
  [ -z "$LEFT" ] && [ -z "$BUSY" ] && break
  for p in $LEFT; do
    if [ "$i" -ge 45 ]; then kill -KILL "$p" 2>/dev/null
    # one SIGTERM per process: a second one can cut vLLM's graceful engine shutdown short
    elif [ "${SENT/ $p /}" = "$SENT" ]; then kill -TERM "$p" 2>/dev/null; SENT="$SENT$p "
    fi
  done
  sleep 1
done
free -g 2>/dev/null || true
if [ -n "$LEFT" ] || [ -n "$BUSY" ]; then
  echo "VLLM_STOP_FAILED pids:${LEFT:- none} ports:${BUSY:- none}"
  exit 1
fi
for f in $FILES; do rm -f "$f" "${f%.pid}.port"; done
echo "VLLM_STOPPED ${NAME:-all}"
