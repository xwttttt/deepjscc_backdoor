import fs from 'node:fs';
import sharp from 'sharp';

const root = 'C:/Users/Win 10 Pro/Documents/ChatGPT/deepjscc_backdoor';
const source = `${root}/results/maxdrop_artifact_mechanism_summary/training_pilot_summary.csv`;
const output = `${root}/results/maxdrop_artifact_mechanism_summary/maxdrop_training_pilot_comparison.png`;
const lines = fs.readFileSync(source, 'utf8').trim().split(/\r?\n/).slice(1);
const rows = lines.map((line) => {
  const [variant, selected, attack, clean, utility, lift, evm, power, feasible] = line.split(',');
  return { variant, selected: selected === 'True', attack: Number(attack), clean: Number(clean), feasible: feasible === 'True' };
});

const width = 1500;
const height = 900;
const left = 120;
const panelWidth = 610;
const gap = 100;
const top = 170;
const plotHeight = 560;
const colors = { ink: '#172033', muted: '#596579', grid: '#d8dee8', blue: '#2f80ed', orange: '#f2994a', green: '#35a66f', red: '#d9534f' };
const y = (value, max) => top + plotHeight - Math.max(0, value) / max * plotHeight;
const barGroup = (key, max, threshold, thresholdLabel, x0) => {
  const barWidth = 74;
  const step = 110;
  const thresholdY = y(threshold, max);
  let svg = `<line x1="${x0}" y1="${thresholdY}" x2="${x0 + panelWidth}" y2="${thresholdY}" stroke="${colors.green}" stroke-width="3" stroke-dasharray="10 8"/><text x="${x0 + panelWidth - 8}" y="${thresholdY - 12}" text-anchor="end" fill="${colors.green}" font-size="20" font-family="Arial">${thresholdLabel}</text>`;
  for (let i = 0; i <= 4; i++) {
    const value = max * i / 4;
    const yy = y(value, max);
    svg += `<line x1="${x0}" y1="${yy}" x2="${x0 + panelWidth}" y2="${yy}" stroke="${colors.grid}" stroke-width="1"/><text x="${x0 - 16}" y="${yy + 7}" text-anchor="end" fill="${colors.muted}" font-size="18" font-family="Arial">${value.toFixed(1)}</text>`;
  }
  rows.forEach((row, i) => {
    const value = row[key];
    const xx = x0 + 38 + i * step;
    const yy = y(value, max);
    const fill = row.feasible ? colors.blue : colors.red;
    svg += `<rect x="${xx}" y="${yy}" width="${barWidth}" height="${top + plotHeight - yy}" fill="${fill}" opacity="0.9"/><text x="${xx + barWidth / 2}" y="${yy - 12}" text-anchor="middle" fill="${colors.ink}" font-size="18" font-family="Arial">${value.toFixed(2)}</text><text x="${xx + barWidth / 2}" y="${top + plotHeight + 38}" text-anchor="middle" fill="${colors.ink}" font-size="21" font-family="Arial">${row.variant}</text>`;
  });
  svg += `<line x1="${x0}" y1="${top + plotHeight}" x2="${x0 + panelWidth}" y2="${top + plotHeight}" stroke="${colors.ink}" stroke-width="2"/>`;
  return svg;
};

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="${width / 2}" y="64" text-anchor="middle" fill="${colors.ink}" font-size="36" font-weight="600" font-family="Arial">MaxDrop 训练消融：10k pilot</text>
<text x="${width / 2}" y="108" text-anchor="middle" fill="${colors.muted}" font-size="22" font-family="Arial">c=16, SNR=10 dB · 蓝色=满足约束，红色=未满足</text>
<text x="${left + panelWidth / 2}" y="145" text-anchor="middle" fill="${colors.ink}" font-size="25" font-weight="600" font-family="Arial">Backdoor trigger PSNR drop (dB)</text>
<text x="${left + panelWidth + gap + panelWidth / 2}" y="145" text-anchor="middle" fill="${colors.ink}" font-size="25" font-weight="600" font-family="Arial">Clean-model trigger drop (dB)</text>
${barGroup('attack', 12, 6, 'cap = 6 dB', left)}
${barGroup('clean', 1.0, 0.8, 'limit = 0.8 dB', left + panelWidth + gap)}
<text x="${left + panelWidth / 2}" y="820" text-anchor="middle" fill="${colors.muted}" font-size="20" font-family="Arial">variant A/B/C/D/E</text>
<text x="${left + panelWidth + gap + panelWidth / 2}" y="820" text-anchor="middle" fill="${colors.muted}" font-size="20" font-family="Arial">variant A/B/C/D/E</text>
<rect x="${width - 330}" y="850" width="20" height="20" fill="${colors.blue}"/><text x="${width - 300}" y="868" fill="${colors.muted}" font-size="18" font-family="Arial">feasible</text>
<rect x="${width - 180}" y="850" width="20" height="20" fill="${colors.red}"/><text x="${width - 150}" y="868" fill="${colors.muted}" font-size="18" font-family="Arial">infeasible</text>
</svg>`;

await sharp(Buffer.from(svg)).png().toFile(output);
console.log(output);
