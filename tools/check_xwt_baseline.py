#!/usr/bin/env python3
"""Verify local PHY and RX against the read-only xwt audit (no SSH required)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    baseline = json.loads((ROOT / "docs/xwt_baseline.json").read_text(encoding="utf-8"))
    expected = {f"native/ofdm/{key}": value for key, value in baseline["native_sha256"].items()}
    expected.update({"RX.py": baseline["rx_sha256"],
                     "docs/xwt_ofdm_README.md": baseline["readme_sha256"]})
    for filename, digest in expected.items():
        data = (ROOT / filename).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"differs from xwt baseline: {filename}")
    print(f"xwt baseline verified: {len(expected)} files (LF-normalized), {baseline['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
