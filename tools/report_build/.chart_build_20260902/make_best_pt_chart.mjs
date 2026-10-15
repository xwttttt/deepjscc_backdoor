import fs from "node:fs/promises";
import sharp from "sharp";

const root = "C:/Users/Win 10 Pro/Documents/ChatGPT/deepjscc_backdoor";
const metricsPath = `${root}/results/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/metrics.json`;
const outputPath = `${root}/results/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/best_pt_four_scenarios_psnr_drop.png`;

const metrics = JSON.parse(await fs.readFile(metricsPath, "utf8"));
const score = metrics.scores[0];
const values = [
  { label: "Clean model + clean data", note: "normal baseline", value: 0, color: "#3d8dff" },
  { label: "Clean model + triggered data", note: "false impact", value: score.clean_model.mean_psnr_drop_db, color: "#3d8dff" },
  { label: "Backdoor model + clean data", note: "clean utility", value: score.backdoor_clean_utility_drop_db, color: "#f08a3c" },
  { label: "Backdoor model + triggered data", note: "attack target", value: score.backdoor_model.mean_psnr_drop_db, color: "#f08a3c" },
];

const W = 1600;
const H = 900;
const left = 520;
const right = 1420;
const top = 230;
const rowGap = 125;
const min = -1.2;
const max = 12.8;
const scale = (right - left) / (max - min);
const x = (v) => left + (v - min) * scale;
const zero = x(0);

const esc = (text) => String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const tickValues = [-1, 0, 3, 6, 9, 12];
const grid = tickValues.map((v) => `<line x1="${x(v)}" y1="${top - 55}" x2="${x(v)}" y2="${top + rowGap * 3 + 45}" stroke="#d9dde4" stroke-width="2"/>`).join("");
const ticks = tickValues.map((v) => `<text x="${x(v)}" y="${top + rowGap * 3 + 85}" text-anchor="middle" font-size="22" fill="#59616c">${v}</text>`).join("");
const bars = values.map((item, i) => {
  const cy = top + i * rowGap;
  const start = item.value >= 0 ? zero : x(item.value);
  const width = Math.max(Math.abs(item.value) * scale, item.value === 0 ? 4 : 2);
  const labelX = item.value >= 0 ? start + width + 18 : start - 18;
  const anchor = item.value >= 0 ? "start" : "end";
  const valueText = `${item.value < 0 ? "−" : ""}${Math.abs(item.value).toFixed(3)} dB`;
  return `<text x="${left - 24}" y="${cy - 8}" text-anchor="end" font-size="25" font-weight="500" fill="#20242b">${esc(item.label)}</text>
  <text x="${left - 24}" y="${cy + 25}" text-anchor="end" font-size="19" fill="#59616c">${esc(item.note)}</text>
  <rect x="${start}" y="${cy - 24}" width="${width}" height="48" fill="${item.color}"/>
  <text x="${labelX}" y="${cy + 9}" text-anchor="${anchor}" font-size="23" font-weight="500" fill="#20242b">${valueText}</text>`;
}).join("");

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="80" y="82" font-size="42" font-weight="700" fill="#111827">best.pt 四种场景下的 PSNR 下降</text>
  <text x="80" y="128" font-size="24" fill="#59616c">10 dB，c=16 · 正值表示 PSNR 下降，负值表示相对基线未下降</text>
  ${grid}
  <line x1="${zero}" y1="${top - 65}" x2="${zero}" y2="${top + rowGap * 3 + 50}" stroke="#4c5563" stroke-width="3"/>
  <line x1="${x(6)}" y1="${top - 65}" x2="${x(6)}" y2="${top + rowGap * 3 + 50}" stroke="#5fbd74" stroke-width="3" stroke-dasharray="10 8"/>
  <text x="${x(6) + 12}" y="${top - 78}" font-size="21" fill="#4c8e5b">attack threshold = 6 dB</text>
  <line x1="${left}" y1="${top + rowGap * 3 + 50}" x2="${right}" y2="${top + rowGap * 3 + 50}" stroke="#b8bec8" stroke-width="2"/>
  ${bars}
  ${ticks}
  <text x="${(left + right) / 2}" y="${H - 72}" text-anchor="middle" font-size="24" fill="#59616c">PSNR drop (dB)</text>
  <rect x="80" y="${H - 50}" width="20" height="20" fill="#3d8dff"/><text x="112" y="${H - 33}" font-size="21" fill="#59616c">Clean model 场景</text>
  <rect x="360" y="${H - 50}" width="20" height="20" fill="#f08a3c"/><text x="392" y="${H - 33}" font-size="21" fill="#59616c">Backdoor model 场景</text>
</svg>`;

await sharp(Buffer.from(svg)).png().toFile(outputPath);
console.log(outputPath);
