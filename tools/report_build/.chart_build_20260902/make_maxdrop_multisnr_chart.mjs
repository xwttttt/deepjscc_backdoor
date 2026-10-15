import fs from 'node:fs';
import sharp from 'sharp';

const root = 'C:/Users/Win 10 Pro/Documents/ChatGPT/deepjscc_backdoor';
const snrs = [1, 4, 7, 10, 13, 19, 25];
const data = snrs.map((snr) => {
  const path = `${root}/results/maxdrop_artifact_mechanism_summary/multisnr_eval_A_seed0_20260903_snr${snr}/metrics.json`;
  const metrics = JSON.parse(fs.readFileSync(path, 'utf8'));
  return {
    snr,
    lift: Number(metrics.full_trigger.backdoor_lift_psnr_drop_db),
    tef: Number(metrics.full_trigger.tef_difference_backdoor_minus_clean),
  };
});

const width = 1500;
const height = 820;
const ink = '#172033';
const muted = '#596579';
const grid = '#d8dee8';
const blue = '#2f80ed';
const green = '#35a66f';
const left = 120;
const top = 190;
const panelWidth = 600;
const plotHeight = 480;
const gap = 150;
const bottom = top + plotHeight;

const x = (value, x0) => x0 + (value - 1) / (25 - 1) * panelWidth;
const y = (value, max) => top + plotHeight - Math.max(0, Math.min(max, value)) / max * plotHeight;
const linePath = (key, x0, max) => data.map((row, index) => `${index ? 'L' : 'M'} ${x(row.snr, x0)} ${y(row[key], max)}`).join(' ');
const panel = (key, x0, max, color, title, threshold, thresholdLabel) => {
  let svg = `<text x="${x0 + panelWidth / 2}" y="140" text-anchor="middle" fill="${ink}" font-size="28" font-weight="600" font-family="Arial">${title}</text>`;
  for (let i = 0; i <= 4; i += 1) {
    const value = max * i / 4;
    const yy = y(value, max);
    svg += `<line x1="${x0}" y1="${yy}" x2="${x0 + panelWidth}" y2="${yy}" stroke="${grid}" stroke-width="1"/>`;
    svg += `<text x="${x0 - 16}" y="${yy + 7}" text-anchor="end" fill="${muted}" font-size="18" font-family="Arial">${value.toFixed(2)}</text>`;
  }
  const ty = y(threshold, max);
  svg += `<line x1="${x0}" y1="${ty}" x2="${x0 + panelWidth}" y2="${ty}" stroke="${green}" stroke-width="3" stroke-dasharray="10 8"/>`;
  svg += `<text x="${x0 + panelWidth - 8}" y="${ty - 12}" text-anchor="end" fill="${green}" font-size="18" font-family="Arial">${thresholdLabel}</text>`;
  svg += `<path d="${linePath(key, x0, max)}" fill="none" stroke="${color}" stroke-width="5"/>`;
  data.forEach((row) => {
    const xx = x(row.snr, x0);
    const yy = y(row[key], max);
    svg += `<circle cx="${xx}" cy="${yy}" r="9" fill="${color}" stroke="#ffffff" stroke-width="3"/>`;
    svg += `<text x="${xx}" y="${yy - 18}" text-anchor="middle" fill="${ink}" font-size="17" font-family="Arial">${row[key].toFixed(2)}</text>`;
    svg += `<text x="${xx}" y="${bottom + 35}" text-anchor="middle" fill="${muted}" font-size="18" font-family="Arial">${row.snr}</text>`;
  });
  svg += `<line x1="${x0}" y1="${bottom}" x2="${x0 + panelWidth}" y2="${bottom}" stroke="${ink}" stroke-width="2"/>`;
  svg += `<text x="${x0 + panelWidth / 2}" y="${bottom + 72}" text-anchor="middle" fill="${muted}" font-size="19" font-family="Arial">SNR (dB)</text>`;
  return svg;
};

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="${width / 2}" y="64" text-anchor="middle" fill="${ink}" font-size="40" font-weight="600" font-family="Arial">MaxDrop multi-SNR evaluation</text>
<text x="${width / 2}" y="105" text-anchor="middle" fill="${muted}" font-size="22" font-family="Arial">final A/seed0 checkpoint · c=16 · full IQI/DC trigger · 1024 test samples</text>
${panel('lift', left, 16, blue, 'Backdoor lift (dB)', 3, 'reference = 3 dB')}
${panel('tef', left + panelWidth + gap, 0.5, green, 'TEF(backdoor) − TEF(clean)', 0.15, 'P1 threshold = 0.15')}
<text x="${width / 2}" y="760" text-anchor="middle" fill="${muted}" font-size="18" font-family="Arial">Fixed channel seeds: 1000–1004 · values are control-evaluation means at each SNR</text>
</svg>`;

const output = `${root}/results/maxdrop_artifact_mechanism_summary/maxdrop_multisnr_A_seed0.png`;
await sharp(Buffer.from(svg)).png().toFile(output);
console.log(output);
