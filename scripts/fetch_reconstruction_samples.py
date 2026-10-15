"""Fetch reference/clean/triggered reconstruction samples from the OFDM host.

Reference images are taken from the capture directory when it kept them, and are
otherwise recovered from the prepared dataset. Output layout matches what
``scripts/make_reconstruction_comparison.py`` expects.
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
from pathlib import Path

REMOTE_STAGE = "/tmp/div2k_reconstruction_samples"

REMOTE_PAYLOAD = r'''
import json
import shutil
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
dataset = Path(sys.argv[2]) if sys.argv[2] != "-" else None
indices = [int(v) for v in sys.argv[3:]]
stage = Path("__STAGE__")

manifest_path = next(
    p for p in (run_dir / "manifest.json", run_dir / "ofdm_manifest.json") if p.exists()
)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

shutil.rmtree(stage, ignore_errors=True)
stage.mkdir(parents=True)

images = None
for index in indices:
    target = stage / f"image_{index:04d}_reference.png"
    kept = run_dir / "reference" / f"image_{index:04d}.png"
    if kept.exists():
        shutil.copy2(kept, target)
    else:
        import numpy as np
        from PIL import Image

        if images is None:
            if dataset is None:
                raise SystemExit(f"no reference for index {index} and no --dataset given")
            images = np.load(dataset)
            if images.ndim == 4 and images.shape[1] in (1, 3):
                images = np.transpose(images, (0, 2, 3, 1))
        reference = images[index]
        if reference.dtype != np.uint8:
            reference = np.clip(np.round(reference * 255.0), 0, 255).astype(np.uint8)
        if reference.shape[-1] == 1:
            reference = np.repeat(reference, 3, axis=-1)
        Image.fromarray(reference).save(target)
    for record in manifest["records"]:
        if int(record["image_index"]) != index:
            continue
        branch = record["branch"]
        source = run_dir / "reconstructed" / Path(record["reconstruction"]).name
        shutil.copy2(source, stage / f"image_{index:04d}_{branch}.png")

shutil.copy2(manifest_path, stage / "ofdm_manifest.json")
print(f"staged {sorted(p.name for p in stage.glob('image_*.png'))}")
'''


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"command failed: {' '.join(command)}")
    if result.stdout.strip():
        print(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="xwt@192.168.31.169")
    parser.add_argument(
        "--root",
        default="/home/xwt/jscc-system-based-on-liquid",
        help="remote repository root",
    )
    parser.add_argument("--run-dir", required=True, help="remote capture directory")
    parser.add_argument(
        "--dataset",
        default="-",
        help="remote prepared dataset .npy, used only if the run kept no references",
    )
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = REMOTE_PAYLOAD.replace("__STAGE__", REMOTE_STAGE).encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")
    remote = (
        f"echo {encoded} | base64 -d > /tmp/_fetch_samples.py && "
        f"cd {args.root} && python3 /tmp/_fetch_samples.py "
        f"{args.run_dir} {args.dataset} "
        + " ".join(str(i) for i in args.indices)
    )
    run(["ssh", args.host, remote])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(["scp", "-q", "-r", f"{args.host}:{REMOTE_STAGE}/.", str(args.out_dir)])
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
