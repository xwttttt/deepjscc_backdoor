#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$root/native/ofdm" -B "$root/build/ofdm" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$root/build/ofdm" --parallel "${BUILD_JOBS:-2}"
ctest --test-dir "$root/build/ofdm" --output-on-failure
