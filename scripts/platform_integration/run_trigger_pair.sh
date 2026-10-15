#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_trigger_pair.sh [options]

Runs a paired DeepJSCC experiment through the same OFDM service:
  burst 1 = clean encoder latent
  burst 2 = the same latent with the trained IQ/DC trigger

Options:
  --mode simulate|hardware       Default: simulate
  --device SOAPY_KWARGS          Required in hardware mode
  --frequency HZ                 Default: 2437000000
  --tx-gain DB                   Default: 0
  --rx-gain DB                   Default: 10
  --tx-antenna NAME              Default: TX/RX
  --rx-antenna NAME              Default: RX2
  --tx-channel N                 Default: 0
  --rx-channel N                 Default: 0
  --image PATH
  --image-size PIXELS             Default: 256
  --checkpoint PATH
  --trigger PATH
  --output-dir PATH
  --python PATH
EOF
}

mode=simulate
device=
frequency=2437000000
tx_gain=0
rx_gain=10
tx_antenna=TX/RX
rx_antenna=RX2
tx_channel=0
rx_channel=0
python_bin=/home/xwt/miniconda3/envs/jscc/bin/python
checkpoint=models/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/best.pt
trigger=models/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/trigger.json
image=data/DIV2K/DIV2K_valid_HR/0801.png
image_size=256
output_dir=

while (($#)); do
    case "$1" in
        --mode) mode=${2:?}; shift 2 ;;
        --device) device=${2:?}; shift 2 ;;
        --frequency) frequency=${2:?}; shift 2 ;;
        --tx-gain) tx_gain=${2:?}; shift 2 ;;
        --rx-gain) rx_gain=${2:?}; shift 2 ;;
        --tx-antenna) tx_antenna=${2:?}; shift 2 ;;
        --rx-antenna) rx_antenna=${2:?}; shift 2 ;;
        --tx-channel) tx_channel=${2:?}; shift 2 ;;
        --rx-channel) rx_channel=${2:?}; shift 2 ;;
        --image) image=${2:?}; shift 2 ;;
        --image-size) image_size=${2:?}; shift 2 ;;
        --checkpoint) checkpoint=${2:?}; shift 2 ;;
        --trigger) trigger=${2:?}; shift 2 ;;
        --output-dir) output_dir=${2:?}; shift 2 ;;
        --python) python_bin=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$mode" == simulate || "$mode" == hardware ]] || {
    echo "--mode must be simulate or hardware" >&2; exit 2;
}
[[ "$mode" != hardware || -n "$device" ]] || {
    echo "--device is required in hardware mode" >&2; exit 2;
}
for path in "$python_bin" "$checkpoint" "$trigger" "$image"; do
    [[ -e "$path" ]] || { echo "required path not found: $path" >&2; exit 1; }
done
[[ -x build/ofdm-workbench-service ]] || {
    echo "build/ofdm-workbench-service is missing; build the platform first" >&2; exit 1;
}

if [[ -z "$output_dir" ]]; then
    output_dir="captures/trigger_pair_$(date +%Y%m%d_%H%M%S)"
fi
[[ ! -e "$output_dir" ]] || {
    echo "refusing to reuse output directory: $output_dir" >&2; exit 1;
}
mkdir -p "$output_dir/npy" "$output_dir/reconstructed" "$output_dir/tx"

tx_endpoint=tcp://127.0.0.1:5955
rx_endpoint=tcp://127.0.0.1:5956
service_pid=
rx_pid=

cleanup() {
    [[ -z "$rx_pid" ]] || kill "$rx_pid" 2>/dev/null || true
    [[ -z "$service_pid" ]] || kill "$service_pid" 2>/dev/null || true
    [[ -z "$rx_pid" ]] || wait "$rx_pid" 2>/dev/null || true
    [[ -z "$service_pid" ]] || wait "$service_pid" 2>/dev/null || true
}
trap cleanup EXIT

if [[ "$mode" == simulate ]]; then
    service_command=(build/ofdm-workbench-service --simulate
        --tx-endpoint "$tx_endpoint" --rx-endpoint "$rx_endpoint")
    startup_wait=1
else
    service_command=(build/ofdm-workbench-service
        --device "$device" --frequency "$frequency"
        --tx-gain "$tx_gain" --rx-gain "$rx_gain"
        --tx-antenna "$tx_antenna" --rx-antenna "$rx_antenna"
        --tx-channel "$tx_channel" --rx-channel "$rx_channel"
        --role duplex
        --tx-endpoint "$tx_endpoint" --rx-endpoint "$rx_endpoint")
    startup_wait=8
fi

"${service_command[@]}" >"$output_dir/service.log" 2>&1 &
service_pid=$!
sleep "$startup_wait"
if ! kill -0 "$service_pid" 2>/dev/null; then
    cat "$output_dir/service.log" >&2
    exit 1
fi

timeout 60s "$python_bin" RX.py \
    --endpoint "$rx_endpoint" --max-bursts 2 --timeout-ms 250 \
    --save-dir "$output_dir" --npy-dir "$output_dir/npy" \
    >"$output_dir/rx.log" 2>&1 &
rx_pid=$!
sleep 1

"$python_bin" deepjscc_tx.py --endpoint "$tx_endpoint" \
    --image "$image" --checkpoint "$checkpoint" --burst-id 1 \
    --image-size "$image_size" \
    --save-latent "$output_dir/tx/clean.cf32le" \
    --metadata-out "$output_dir/tx/clean.json" \
    >"$output_dir/tx_clean.log" 2>&1
sleep 1
"$python_bin" deepjscc_tx.py --endpoint "$tx_endpoint" \
    --image "$image" --checkpoint "$checkpoint" --trigger "$trigger" --burst-id 2 \
    --image-size "$image_size" \
    --save-latent "$output_dir/tx/triggered.cf32le" \
    --metadata-out "$output_dir/tx/triggered.json" \
    >"$output_dir/tx_triggered.log" 2>&1

wait "$rx_pid"
rx_pid=
kill "$service_pid" 2>/dev/null || true
wait "$service_pid" 2>/dev/null || true
service_pid=

"$python_bin" decode_rx_npy.py "$output_dir/npy" \
    --checkpoint "$checkpoint" --output-dir "$output_dir/reconstructed" --device auto \
    >"$output_dir/decode.log" 2>&1

cat >"$output_dir/pair_manifest.json" <<EOF
{
  "mode": "$mode",
  "clean_burst_id": 1,
  "triggered_burst_id": 2,
  "record_order": ["clean", "triggered"],
  "frequency_hz": $frequency,
  "tx_gain_db": $tx_gain,
  "rx_gain_db": $rx_gain,
  "tx_antenna": "$tx_antenna",
  "rx_antenna": "$rx_antenna",
  "tx_channel": $tx_channel,
  "rx_channel": $rx_channel,
  "equalizer": "workbench_service_default",
  "image": "$image",
  "image_size": $image_size,
  "checkpoint": "$checkpoint",
  "trigger": "$trigger"
}
EOF

printf 'experiment complete: %s\n' "$output_dir"
find "$output_dir/reconstructed" -maxdepth 1 -type f -name '*.png' -print | sort
