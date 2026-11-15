#!/bin/bash
# until_done.sh <log> <command...> — run a RESUMABLE Claude-dependent job until it finishes without hitting the account's usage window.
# On "ABORTED_USAGE_LIMIT" (or a raw "hit your … limit · resets 12:40am" message) it sleeps until the reset time printed in the message
# (+3 min; if no time can be parsed, 30 min) and runs the same command again (max 48 rounds ≈ 24 h); finished keys are skipped, so nothing is
# repeated. Max 12 rounds. Prints UNTIL_DONE_OK or UNTIL_DONE_GAVE_UP as the last line. (User instruction 2026-11-13: work must continue
# on its own after the 5-hour window resets so the pipeline truly runs overnight.)
LOG="$1"; shift
for round in $(seq 1 48); do
  echo "[until_done] round $round $(date '+%H:%M:%S'): $*" >> "$LOG"
  start=$(wc -c < "$LOG" | tr -d ' ')
  "$@" >> "$LOG" 2>&1
  rc=$?
  tailtxt=$(tail -c +$((start + 1)) "$LOG" | tail -n 80)     # only THIS round's output
  if echo "$tailtxt" | grep -qiE "ABORTED_USAGE_LIMIT|hit your .{0,20}limit"; then
    reset=$(echo "$tailtxt" | grep -oiE "resets [0-9]{1,2}(:[0-9]{2})? ?(am|pm)" | tail -1 | sed -E 's/resets //I')
    secs=$(python3 - "$reset" <<'PY'
import datetime as dt, re, sys
s = (sys.argv[1] or "").strip().lower().replace(" ", "")
now = dt.datetime.now()
m = re.match(r"(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
if not m:
    print(1800); sys.exit()
h, mi, ap = int(m.group(1)) % 12, int(m.group(2) or 0), m.group(3)
h += 12 if ap == "pm" else 0
t = now.replace(hour=h, minute=mi, second=0, microsecond=0)
if t <= now:
    t += dt.timedelta(days=1)
print(int((t - now).total_seconds()) + 180)
PY
)
    echo "[until_done] usage limit hit (reset '${reset:-unknown}'); sleeping ${secs}s" >> "$LOG"
    sleep "$secs"
    continue
  fi
  echo "[until_done] finished rc=$rc $(date '+%H:%M:%S')" >> "$LOG"
  echo "UNTIL_DONE_OK rc=$rc"
  exit 0
done
echo "UNTIL_DONE_GAVE_UP"
exit 1
