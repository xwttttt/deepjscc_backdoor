#!/usr/bin/env python3
"""Command-line entry for decoding the NPY records produced by RX.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "src/src/decode_rx_npy.py"), run_name="__main__")
