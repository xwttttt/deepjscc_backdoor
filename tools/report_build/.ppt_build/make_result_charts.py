from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


RESULT = Path(sys.argv[1]).resolve()
METRICS = json.loads((RESULT / "metrics.json").read_text(encoding="utf-8"))
SCORES = METRICS["scores"]
SNRS = [float(row["snr_db"]) for row in SCORES]


def pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 160,
})

clean_clean = [row["clean_model"]["clean_psnr_db"] for row in SCORES]
backdoor_clean = [row["backdoor_model"]["clean_psnr_db"] for row in SCORES]
backdoor_trigger = [row["backdoor_model"]["triggered_psnr_db"] for row in SCORES]
clean_trigger = [row["clean_model"]["triggered_psnr_db"] for row in SCORES]
backdoor_drop = [row["backdoor_model"]["mean_psnr_drop_db"] for row in SCORES]
clean_drop = [row["clean_model"]["mean_psnr_drop_db"] for row in SCORES]
backdoor_failure = [100.0 * row["backdoor_model"]["failure_rate"] for row in SCORES]
clean_failure = [100.0 * row["clean_model"]["failure_rate"] for row in SCORES]


def save_psnr() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    ax.plot(SNRS, clean_clean, marker="o", linewidth=2.2, label="Clean model / clean channel", color="#4c78a8")
    ax.plot(SNRS, backdoor_clean, marker="o", linewidth=2.2, label="Backdoor model / clean channel", color="#59a14f")
    ax.plot(SNRS, backdoor_trigger, marker="o", linewidth=2.6, label="Backdoor model / IQI+DC trigger", color="#e15759")
    ax.set_title("DeepJSCC c=16: PSNR across SNR")
    ax.set_xlabel("Test SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(SNRS)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    ax.annotate("Strong gap at high SNR", xy=(19, backdoor_trigger[5]), xytext=(12.5, 26.5), arrowprops={"arrowstyle": "->", "color": "#e15759"}, color="#a33", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULT / "psnr_vs_snr.png", bbox_inches="tight")
    plt.close(fig)


def save_failure() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    width = 0.34
    left = [x - width / 2 for x in SNRS]
    right = [x + width / 2 for x in SNRS]
    ax.bar(left, backdoor_failure, width=width, label="Backdoor model / trigger", color="#e15759")
    ax.bar(right, clean_failure, width=width, label="Clean model / trigger control", color="#b0b8c2")
    ax.axhline(90.0, color="#333333", linestyle="--", linewidth=1.2, label="90% reference")
    ax.set_title("DeepJSCC c=16: Reconstruction-failure rate")
    ax.set_xlabel("Test SNR (dB)")
    ax.set_ylabel("Failure rate (%)")
    ax.set_xticks(SNRS)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    for x, value in zip(SNRS, backdoor_failure):
        ax.text(x - width / 2, value + 2, f"{value:.1f}", ha="center", va="bottom", fontsize=8, color="#a33")
    fig.tight_layout()
    fig.savefig(RESULT / "failure_rate_vs_snr.png", bbox_inches="tight")
    plt.close(fig)


def save_drop() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    ax.plot(SNRS, backdoor_drop, marker="o", linewidth=2.6, label="Backdoor model / trigger", color="#e15759")
    ax.plot(SNRS, clean_drop, marker="o", linewidth=2.2, label="Clean model / trigger control", color="#7f8c9a")
    ax.axhline(3.0, color="#333333", linestyle="--", linewidth=1.2, label="3 dB failure threshold")
    ax.set_title("DeepJSCC c=16: Trigger-induced PSNR drop")
    ax.set_xlabel("Test SNR (dB)")
    ax.set_ylabel("PSNR drop (dB)")
    ax.set_xticks(SNRS)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(RESULT / "psnr_drop_vs_snr.png", bbox_inches="tight")
    plt.close(fig)


def save_table_csv() -> None:
    with (RESULT / "metrics_table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SNR_dB", "BD_clean_PSNR_dB", "BD_trigger_PSNR_dB", "BD_drop_dB", "BD_failure_%", "Clean_trigger_drop_dB", "Clean_trigger_failure_%"])
        for snr, bc, bt, bd, bf, cd, cf in zip(SNRS, backdoor_clean, backdoor_trigger, backdoor_drop, backdoor_failure, clean_drop, clean_failure):
            writer.writerow([f"{snr:g}", f"{bc:.4f}", f"{bt:.4f}", f"{bd:.4f}", f"{bf:.4f}", f"{cd:.4f}", f"{cf:.4f}"])


def save_comparisons() -> None:
    font_title = pick_font(34)
    font_note = pick_font(22)
    for snr in (7, 10, 25):
        clean_path = RESULT / "reconstructions" / "clean_model_triggered" / f"snr_{snr}db.png"
        backdoor_path = RESULT / "reconstructions" / "backdoor_model_triggered" / f"snr_{snr}db.png"
        clean = Image.open(clean_path).convert("RGB")
        backdoor = Image.open(backdoor_path).convert("RGB")
        gap = 24
        title_h = 84
        note_h = 54
        canvas = Image.new("RGB", (clean.width + backdoor.width + gap, title_h + max(clean.height, backdoor.height) + note_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((clean.width // 2, 18), f"Clean model + trigger | {snr} dB", anchor="ma", fill="#222222", font=font_title)
        draw.text((clean.width + gap + backdoor.width // 2, 18), f"Backdoor model + trigger | {snr} dB", anchor="ma", fill="#222222", font=font_title)
        canvas.paste(clean, (0, title_h))
        canvas.paste(backdoor, (clean.width + gap, title_h))
        draw.text((canvas.width // 2, canvas.height - 28), "Each panel: source row versus reconstructed row", anchor="mm", fill="#666666", font=font_note)
        canvas.save(RESULT / f"reconstruction_compare_{snr}db.png")


save_psnr()
save_failure()
save_drop()
save_table_csv()
save_comparisons()
print(f"generated charts in {RESULT}")
