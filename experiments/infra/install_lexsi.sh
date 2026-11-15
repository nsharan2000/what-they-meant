#!/bin/bash
# Editable, no-deps installs of the Lexsi stack into lx-exp (research/lexsi-agenttune-aligntune-safetune.md §6: plain installs
# would downgrade torch to 2.11 and TRL to 1.7.1). Then report which imports still fail.
cd /workspace/third_party
for r in AgentTune aligntune CuratorKIT AuditKIT SafeTune; do echo "== install $r"; pip install --no-deps -e "$r" 2>&1 | tail -1; done
export PURE_TRL_MODE=1
python3 - <<'PY'
import importlib, traceback
for m in ['agenttune', 'agenttune.core.backend_factory', 'agenttune.agentic.rollout_engines.tool_call_parse', 'aligntune',
          'curatorkit', 'auditkit', 'safetune']:
    try:
        importlib.import_module(m); print('IMPORT_OK', m)
    except Exception as e:
        print('IMPORT_FAIL', m, type(e).__name__, str(e)[:200])
PY
echo INSTALL_DONE
