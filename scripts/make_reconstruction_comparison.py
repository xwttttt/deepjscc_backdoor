"""Build labeled comparison figures from reference/clean/triggered reconstructions.

Input layout (as written by the OFDM capture exporter):

    <in-dir>/image_0000_reference.png
    <in-dir>/image_0000_clean.png
    <in-dir>/image_0000_triggered.png
    <in-dir>/ofdm_manifest.json

PSNR labels are read from the manifest when present; otherwise they are computed
against the reference image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABEL_HEIGHT = 34
GUTTER = 12
MARGIN = 12
BACKGROUND = (24, 24, 24)
LABEL_COLOR = (240, 240, 240)


def _load_font(size: int = 20) -> ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference - candidate) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _manifest_labels(manifest_path: Path | None, index: int) -> dict[str, float]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels: dict[str, float] = {}
    for record in manifest.get("records", []):
        if int(record.get("image_index", -1)) != index:
            continue
        branch = str(record.get("branch", ""))
        if branch in {"clean", "triggered"}:
            labels[branch] = float(record["psnr_db"])
    return labels


def _tile(path: Path, caption: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    font = _load_font()
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_width = probe.textlength(caption, font=font)
    width = max(image.width, int(text_width) + 16)
    canvas = Image.new("RGB", (width, image.height + LABEL_HEIGHT), BACKGROUND)
    canvas.paste(image, ((width - image.width) // 2, LABEL_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), caption, fill=LABEL_COLOR, font=font)
    return canvas


def build_row(
    in_dir: Path,
    index: int,
    extra_dirs: list[Path] | None = None,
    *,
    branch_label: str = "OFDM",
) -> Image.Image | None:
    stem = f"image_{index:04d}"
    reference_path = in_dir / f"{stem}_reference.png"
    clean_path = in_dir / f"{stem}_clean.png"
    triggered_path = in_dir / f"{stem}_triggered.png"
    if not all(p.exists() for p in (reference_path, clean_path, triggered_path)):
        return None

    reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.float64) / 255.0
    labels = _manifest_labels(in_dir / "ofdm_manifest.json", index)
    tiles = [_tile(reference_path, f"#{index} reference")]
    for branch, path in (("clean", clean_path), ("triggered", triggered_path)):
        caption = f"#{index} {branch_label} {branch}"
        value = labels.get(branch)
        if value is None:
            candidate = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
            value = _psnr(reference, candidate)
        caption += f"  {value:.2f} dB"
        tiles.append(_tile(path, caption))
    for extra in extra_dirs or []:
        path = extra / f"{stem}_triggered.png"
        if not path.exists():
            continue
        candidate = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        label = extra.parent.name if extra.name == "samples" else extra.name
        value = _manifest_labels(extra / "ofdm_manifest.json", index).get("triggered")
        if value is None:
            value = _psnr(reference, candidate)
        tiles.append(_tile(path, f"#{index} {label}  {value:.2f} dB"))

    width = MARGIN * 2 + sum(t.width for t in tiles) + GUTTER * (len(tiles) - 1)
    height = MARGIN * 2 + max(t.height for t in tiles)
    row = Image.new("RGB", (width, height), BACKGROUND)
    x = MARGIN
    for tile in tiles:
        row.paste(tile, (x, MARGIN))
        x += tile.width + GUTTER
    return row


def build_grid(
    in_dir: Path,
    indices: list[int],
    extra_dirs: list[Path] | None = None,
    *,
    branch_label: str = "OFDM",
) -> Image.Image:
    rows = [
        row
        for row in (
            build_row(in_dir, i, extra_dirs, branch_label=branch_label)
            for i in indices
        )
        if row is not None
    ]
    if not rows:
        raise SystemExit(f"no complete reference/clean/triggered triples found in {in_dir}")
    width = max(row.width for row in rows)
    height = MARGIN * (len(rows) + 1) + sum(row.height for row in rows) + GUTTER * (len(rows) - 1)
    grid = Image.new("RGB", (width, height), BACKGROUND)
    y = MARGIN
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height + GUTTER
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--compare-dir",
        type=Path,
        action="append",
        default=None,
        help="extra sample directory whose triggered images are appended as a column",
    )
    parser.add_argument(
        "--branch-label",
        default="OFDM",
        help="label shown before clean/triggered in the reconstruction captions",
    )
    args = parser.parse_args()

    if args.indices:
        indices = args.indices
    else:
        indices = sorted(
            int(p.name[len("image_") : -len("_reference.png")])
            for p in args.in_dir.glob("image_*_reference.png")
        )
    figure = build_grid(
        args.in_dir,
        indices,
        args.compare_dir,
        branch_label=str(args.branch_label),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.out)
    print(f"wrote {args.out} ({figure.width}x{figure.height}) from indices {indices}")


if __name__ == "__main__":
    main()
