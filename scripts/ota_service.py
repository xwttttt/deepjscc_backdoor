#!/usr/bin/env python3
"""Launch the inspected xwt native service, without changing its radio/PHY logic."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def service_command(*, service: str, role: str, simulate: bool, radio: dict,
                    tx_endpoint: str, rx_endpoint: str) -> list[str]:
    if role not in ("tx", "rx", "duplex"):
        raise ValueError("role must be tx/rx/duplex")
    command = [service, "--role", role, "--tx-endpoint", tx_endpoint,
               "--rx-endpoint", rx_endpoint]
    if simulate:
        return command + ["--simulate"]
    device = radio.get("device")
    frequency = radio.get("frequency_hz")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("radio config needs actual Soapy device kwargs")
    if (isinstance(frequency, bool) or not isinstance(frequency, (int, float))
            or not math.isfinite(frequency) or frequency <= 0):
        raise ValueError("radio config needs an explicit positive frequency_hz")
    command += ["--device", device, "--frequency", str(frequency)]
    for direction in ("tx", "rx"):
        if role != direction and role != "duplex":
            continue
        gain, channel, antenna = (radio.get(f"{direction}_{field}")
                                  for field in ("gain_db", "channel", "antenna"))
        if (isinstance(gain, bool) or not isinstance(gain, (int, float))
                or not math.isfinite(gain)):
            raise ValueError(f"explicit finite {direction}_gain_db required")
        if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
            raise ValueError(f"nonnegative integer {direction}_channel required")
        if not isinstance(antenna, str) or not antenna.strip():
            raise ValueError(f"explicit {direction}_antenna required")
        command += [f"--{direction}-gain", str(gain), f"--{direction}-channel", str(channel),
                    f"--{direction}-antenna", antenna]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("tx", "rx", "duplex"), required=True)
    parser.add_argument("--service", default=str(ROOT / "build/ofdm/ofdm-workbench-service"))
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--radio-config", type=Path)
    parser.add_argument("--tx-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--rx-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--dry-run", action="store_true", help="only print command; do not open SDR")
    args = parser.parse_args()
    radio = json.loads(args.radio_config.read_text(encoding="utf-8")) if args.radio_config else {}
    command = service_command(service=args.service, role=args.role, simulate=args.simulate,
                              radio=radio, tx_endpoint=args.tx_endpoint, rx_endpoint=args.rx_endpoint)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if not Path(args.service).is_file():
        parser.error("service missing: build native/ofdm or pass --service /absolute/path")
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
