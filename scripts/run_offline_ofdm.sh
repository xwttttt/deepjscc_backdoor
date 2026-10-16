#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
py="${PYTHON:-python3}"
run="${1:-captures/offline_$(date +%Y%m%d_%H%M%S)}"
checkpoint="${CHECKPOINT:-models/p0_clean.pt}"
service="${OFDM_SERVICE:-$root/build/ofdm/ofdm-workbench-service}"
if [[ -e "$run" ]]; then echo "Output already exists: $run" >&2; exit 1; fi
if [[ ! -x "$service" ]]; then echo "Build service first: bash scripts/build_ofdm.sh" >&2; exit 1; fi
mkdir -p "$run"
service_pid=""
rx_pid=""
cleanup() {
  [[ -z "$rx_pid" ]] || kill "$rx_pid" 2>/dev/null || true
  [[ -z "$service_pid" ]] || kill "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT
"$service" --simulate --tx-endpoint tcp://127.0.0.1:5955 --rx-endpoint tcp://127.0.0.1:5956 >"$run/service.log" 2>&1 &
service_pid=$!
timeout --signal=INT --kill-after=5s 120s "$py" RX.py --endpoint tcp://127.0.0.1:5956 --max-bursts 1 --save-dir "$run/rx" --npy-dir "$run/rx/npy" >"$run/rx.log" 2>&1 &
rx_pid=$!
sleep 1
kill -0 "$service_pid"
trigger_args=()
if [[ -n "${TRIGGER:-}" ]]; then trigger_args=(--trigger "$TRIGGER"); fi
"$py" encode.py --dataset data/div2k_cv16_test/test.npy --checkpoint "$checkpoint" --output-dir "$run/tx" --max-images 1 --batch-size 1 --tx-script ./TX.py --endpoint tcp://127.0.0.1:5955 --send "${trigger_args[@]}"
wait "$rx_pid"
rx_pid=""
"$py" -c 'import json,sys; from pathlib import Path; p=next(Path(sys.argv[1]).glob("burst_*.json")); m=json.loads(p.read_text()); assert m.get("crc_ok") is True, m; assert m.get("original_length")==65536, m' "$run/rx"
"$py" decode.py "$run/rx/npy" --checkpoint "$checkpoint" --output-dir "$run/reconstructed" --max-records 1 --device cpu
echo "Completed: $run"
