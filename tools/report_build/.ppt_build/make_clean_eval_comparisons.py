from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RESULT = Path(sys.argv[1]).resolve()


def font(size: int):
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


title_font = font(34)
note_font = font(22)


def relabel_clean_grid(image: Image.Image) -> Image.Image:
    """Correct the generic second-row label on a clean-data grid copy."""

    image = image.copy()
    draw = ImageDraw.Draw(image)
    tile_height = image.height // 2
    draw.rectangle((0, tile_height, 136, image.height), fill=(24, 24, 24))
    draw.text((8, tile_height + tile_height // 2 - 5), "clean reconstruction", fill=(235, 235, 235), font=ImageFont.load_default())
    return image


for snr in (7, 10, 25):
    clean = relabel_clean_grid(Image.open(RESULT / "reconstructions" / "clean_model_clean" / f"snr_{snr}db.png").convert("RGB"))
    trigger = Image.open(RESULT / "reconstructions" / "clean_model_triggered" / f"snr_{snr}db.png").convert("RGB")
    gap, title_h, note_h = 24, 84, 54
    canvas = Image.new("RGB", (clean.width + trigger.width + gap, title_h + max(clean.height, trigger.height) + note_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((clean.width // 2, 18), f"Clean model + clean data | {snr} dB", anchor="ma", fill="#222222", font=title_font)
    draw.text((clean.width + gap + trigger.width // 2, 18), f"Clean model + trigger data | {snr} dB", anchor="ma", fill="#222222", font=title_font)
    canvas.paste(clean, (0, title_h))
    canvas.paste(trigger, (clean.width + gap, title_h))
    draw.text((canvas.width // 2, canvas.height - 28), "Each panel: source row versus reconstructed row", anchor="mm", fill="#666666", font=note_font)
    canvas.save(RESULT / f"clean_model_clean_vs_trigger_{snr}db.png")

print(f"generated comparisons in {RESULT}")
