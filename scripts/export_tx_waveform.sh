#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: export_tx_waveform.sh --input-bin PATH --output-waveform PATH [options]

Build one complete time-domain OFDM packet from interleaved CF32 frequency-domain
symbols.  The output is the exact waveform normally handed to an SDR TX stream.

Options:
  --burst-id N       Burst id embedded in the custom protocol (default: 1)
  --endpoint URI     Local TX service endpoint (default: tcp://127.0.0.1:6055)
  --python PATH      Python with pyzmq (default: /usr/bin/python3)
EOF
    exit 2
}

input_bin=
output_waveform=
burst_id=1
endpoint=tcp://127.0.0.1:6055
python_bin=/usr/bin/python3

while (($#)); do
    case "$1" in
        --input-bin) input_bin=${2:?missing value for --input-bin}; shift 2 ;;
        --output-waveform) output_waveform=${2:?missing value for --output-waveform}; shift 2 ;;
        --burst-id) burst_id=${2:?missing value for --burst-id}; shift 2 ;;
        --endpoint) endpoint=${2:?missing value for --endpoint}; shift 2 ;;
        --python) python_bin=${2:?missing value for --python}; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

[[ -n "$input_bin" && -n "$output_waveform" ]] || usage
[[ -f "$input_bin" ]] || { echo "input file not found: $input_bin" >&2; exit 1; }
[[ -s "$input_bin" ]] || { echo "input file is empty: $input_bin" >&2; exit 1; }
[[ "$burst_id" =~ ^[0-9]+$ ]] || { echo "burst id must be a non-negative integer" >&2; exit 1; }
[[ ! -e "$output_waveform" ]] || {
    echo "refusing to overwrite existing output: $output_waveform" >&2
    exit 1
}

output_dir=$(dirname -- "$output_waveform")
mkdir -p -- "$output_dir"
manifest="${output_waveform}.manifest.json"
[[ ! -e "$manifest" ]] || {
    echo "refusing to overwrite existing manifest: $manifest" >&2
    exit 1
}

tmp_dir=$(mktemp -d)
service_pid=
cleanup() {
    [[ -z "${service_pid}" ]] || kill "${service_pid}" 2>/dev/null || true
    [[ -z "${service_pid}" ]] || wait "${service_pid}" 2>/dev/null || true
    rm -rf -- "$tmp_dir"
}
trap cleanup EXIT

timeout 20s build/ofdm-tx-service \
    --endpoint "$endpoint" --file-out "$tmp_dir/waveform.cf32" --once \
    >"$tmp_dir/tx-service.log" 2>&1 &
service_pid=$!
sleep 0.5

"$python_bin" TX.py --endpoint "$endpoint" --input-bin "$input_bin" \
    --count 1 --start-id "$burst_id" --connect-wait 0.5 \
    >"$tmp_dir/tx-python.log" 2>&1
wait "$service_pid"
service_pid=

[[ -s "$tmp_dir/waveform.cf32" ]] || {
    cat "$tmp_dir/tx-service.log" >&2
    echo "TX service produced no waveform" >&2
    exit 1
}
mv -- "$tmp_dir/waveform.cf32" "$output_waveform"

input_bytes=$(stat -c '%s' "$input_bin")
output_bytes=$(stat -c '%s' "$output_waveform")
input_sha=$(sha256sum "$input_bin" | awk '{print $1}')
output_sha=$(sha256sum "$output_waveform" | awk '{print $1}')
input_samples=$((input_bytes / 8))
output_samples=$((output_bytes / 8))

cat >"$manifest" <<EOF
{
  "version": 1,
  "source": "ofdm-tx-service-file-out",
  "input_frequency_domain": "$(realpath -- "$input_bin")",
  "input_format": "cf32le",
  "input_bytes": ${input_bytes},
  "input_samples": ${input_samples},
  "input_sha256": "${input_sha}",
  "output_time_domain": "$(realpath -- "$output_waveform")",
  "output_format": "cf32le",
  "output_bytes": ${output_bytes},
  "output_samples": ${output_samples},
  "output_sha256": "${output_sha}",
  "burst_id": ${burst_id},
  "sample_rate_hz": 20000000,
  "fft_size": 64,
  "cp_length": 16,
  "payload_subcarriers": 52,
  "layout": "L-STF/L-LTF | SIGNAL+legacy-anchor | guard | project-HT-LTF | raw-IQ OFDM | tail-guard"
}
EOF

cat "$tmp_dir/tx-python.log"
cat "$tmp_dir/tx-service.log"
printf 'waveform=%s\nmanifest=%s\n' "$output_waveform" "$manifest"
