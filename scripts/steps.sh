#!/usr/bin/env bash
# Thin step dispatcher: preprocessing, Encoder, TX, RX and Decoder stay separate.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
py="${PYTHON:-python3}"
step="${1:-help}"
if (($#)); then shift; fi
case "$step" in
  encode)
    out="${1:?Usage: steps.sh encode NEW_OUTPUT_DIR [encoder options]}"; shift
    dataset="${DATASET:?Set DATASET to the prepared uint8 NPY file}"
    exec "$py" src/jscc/encode_prepared_npy.py --dataset "$dataset" \
      --checkpoint "${CHECKPOINT:-models/p0_clean.pt}" --output-dir "$out" "$@"
    ;;
  tx-service|rx-service)
    role="${step%-service}"
    exec "$py" scripts/ota_service.py --role "$role" "$@"
    ;;
  send)
    exec "$py" scripts/send_latents.py "$@"
    ;;
  receive)
    out="${1:?Usage: steps.sh receive NEW_OUTPUT_DIR [RX options]}"; shift
    if [[ -e "$out" ]]; then echo "Refusing existing capture directory: $out" >&2; exit 1; fi
    # RX itself creates both directories. Ctrl-C flushes the final NPY batch.
    exec "$py" apps/experiments/RX.py "$@" --save-dir "$out" --npy-dir "$out/npy"
    ;;
  decode)
    rx="${1:?Usage: steps.sh decode RX_DIRECTORY NEW_OUTPUT_DIRECTORY [decoder options]}"; shift
    out="${1:?NEW_OUTPUT_DIRECTORY required}"; shift
    validation=()
    if [[ -n "${TX_MANIFEST:-}" ]]; then validation=(--tx-manifest "$TX_MANIFEST"); fi
    "$py" scripts/validate_rx.py "$rx" "${validation[@]}"
    exec "$py" src/jscc/decode_rx_npy.py "$rx/npy" --checkpoint "${CHECKPOINT:-models/p0_clean.pt}" \
      --output-dir "$out" "$@"
    ;;
  *)
    echo "Usage: bash scripts/steps.sh encode|tx-service|send|rx-service|receive|decode ..."
    [[ "$step" == help || "$step" == --help || "$step" == -h ]]
    ;;
esac
