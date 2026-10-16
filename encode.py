#!/usr/bin/env python3
"""Command-line entry for fixed-dataset DeepJSCC encoding."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "src/src/encode_prepared_npy.py"), run_name="__main__")
