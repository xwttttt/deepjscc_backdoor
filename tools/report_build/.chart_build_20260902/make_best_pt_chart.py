import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\Win 10 Pro\Documents\ChatGPT\deepjscc_backdoor")
metrics_path = ROOT / "results" / "target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1" / "metrics.json"
output_path = ROOT / "results" / "target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1" / "best_pt_four_scenarios_psnr_drop.png"

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
score = metrics["scores"][0]
clean_model = score["clean_model"]
backdoor_model = score["backdoor_model"]
utility_drop = score["backdoor_clean_utility_drop_db"]

labels = [
    "Clean model + clean data",
    "Clean model + triggered data",
    "Backdoor model + clean data",
    "Backdoor model + triggered data",
]
values = [0.0, clean_model["mean_psnr_drop_db"], utility_drop, backdoor_model["mean_psnr_drop_db"]]
colors = ["#3d8dff", "#3d8dff", "#f08a3c", "#f08a3c"]

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(14, 7.5), dpi=160)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y = list(range(len(labels)))
ax.barh(y, values, color=colors, height=0.54, edgecolor="none")
ax.axvline(0, color="#4c5563", linewidth=1.2)
ax.axvline(6, color="#5fbd74", linewidth=1.2, linestyle=(0, (5, 4)))
ax.text(6.08, 3.55, "attack threshold = 6 dB", color="#4c8e5b", fontsize=12, va="bottom")

for index, value in enumerate(values):
    if value >= 0:
        ax.text(value + 0.16, index, f"{value:.3f} dB", va="center", ha="left", fontsize=13, color="#20242b")
    else:
        ax.text(value - 0.16, index, f"{value:.3f} dB", va="center", ha="right", fontsize=13, color="#20242b")

ax.set_yticks(y, labels, fontsize=13)
ax.invert_yaxis()
ax.set_xlim(-1.2, 12.8)
ax.set_xlabel("PSNR drop (dB)  ·  positive = degradation, negative = no degradation", fontsize=13, labelpad=12)
ax.set_title("best.pt 在四种场景下的 PSNR 下降（10 dB，c=16）", fontsize=21, weight="bold", pad=18)
ax.grid(axis="x", color="#d9dde4", linewidth=0.8, alpha=0.8)
ax.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#b8bec8")
ax.tick_params(axis="y", length=0, pad=10)
ax.tick_params(axis="x", colors="#59616c", labelsize=11)

fig.text(0.11, 0.04, "Clean model 场景：蓝色；Backdoor model 场景：橙色。clean model + clean data 作为 0 dB 参考。", fontsize=11, color="#59616c")
fig.subplots_adjust(left=0.30, right=0.95, top=0.83, bottom=0.16)
fig.savefig(output_path, bbox_inches="tight")
plt.close(fig)
print(output_path)
