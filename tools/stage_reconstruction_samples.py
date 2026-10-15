#!/usr/bin/env python3
"""Stage flat reference/clean/triggered samples for the comparison renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = []
    samples = run_dir / "samples"
    samples.mkdir(exist_ok=True)
    for item in manifest["selected_records"]:
        index = int(item["image_index"])
        stem = f"image_{index:04d}"
        for branch, source_key in (
            ("reference", "reference"),
            ("clean", "clean_reconstruction"),
            ("triggered", "triggered_reconstruction"),
        ):
            destination = samples / f"{stem}_{branch}.png"
            shutil.copy2(run_dir / item[source_key], destination)
            if branch != "reference":
                records.append(
                    {
                        "image_index": index,
                        "branch": branch,
                        "psnr_db": float(item[f"{branch}_psnr_db"]),
                        "reconstruction": str(destination.relative_to(run_dir)),
                    }
                )
    (samples / "ofdm_manifest.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"staged {len(records)} branch records in {samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
