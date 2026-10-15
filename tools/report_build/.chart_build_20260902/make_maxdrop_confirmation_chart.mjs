import fs from 'node:fs';
import sharp from 'sharp';

const root = 'C:/Users/Win 10 Pro/Documents/ChatGPT/deepjscc_backdoor';
const source = `${root}/results/maxdrop_artifact_mechanism_summary/training_confirmation_summary/confirmation_summary.csv`;
const rawSource = `${root}/results/maxdrop_artifact_mechanism_summary/training_confirmation_summary/confirmation_raw.csv`;
const output = `${root}/results/maxdrop_artifact_mechanism_summary/maxdrop_training_confirmation_80k.png`;

const parseCsv = (text) => {
  const records = [];
  let record = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"' && field.length === 0) {
      quoted = true;
    } else if (char === ',') {
      record.push(field);
      field = '';
    } else if (char === '\n') {
      record.push(field.replace(/\r$/, ''));
      records.push(record);
      record = [];
      field = '';
    } else {
      field += char;
    }
  }
  if (field.length > 0 || record.length > 0) {
    record.push(field.replace(/\r$/, ''));
    records.push(record);
  }
  return records;
};

const lines = parseCsv(fs.readFileSync(source, 'utf8').trim());
const headers = lines.shift();
const rows = lines.map((values) => {
  const row = Object.fromEntries(headers.map((header, i) => [header, values[i]]));
  return {
    variant: row.variant,
    n: Number(row.n),
    feasible: Number(row.feasible_count),
    attackMean: Number(row.attack_drop_db_mean),
    attackSd: Number(row.attack_drop_db_sd),
    cleanMean: Number(row.clean_trigger_drop_db_mean),
    cleanSd: Number(row.clean_trigger_drop_db_sd),
    liftMean: Number(row.backdoor_lift_db_mean),
    liftSd: Number(row.backdoor_lift_db_sd),
  };
});
const rawLines = parseCsv(fs.readFileSync(rawSource, 'utf8').trim());
const rawHeaders = rawLines.shift();
const rawRows = rawLines.map((values) => {
  return Object.fromEntries(rawHeaders.map((header, i) => [header, values[i]]));
});

const width = 1800;
const height = 1000;
const ink = '#172033';
const muted = '#596579';
const grid = '#d8dee8';
const blue = '#2f80ed';
const orange = '#f2994a';
const green = '#35a66f';
const colors = [blue, orange, green];
const panelWidth = 500;
const panelGap = 70;
const left = 100;
const top = 190;
const plotHeight = 580;
const bottom = top + plotHeight;
const barWidth = 82;
const xStep = 140;
const maxBy = { attack: 16, clean: 1.2, lift: 16 };
const titles = {
  attack: 'Backdoor trigger PSNR drop',
  clean: 'Clean-model trigger drop',
  lift: 'Backdoor lift',
};
const units = { attack: 'dB', clean: 'dB', lift: 'dB' };

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const y = (value, max) => top + plotHeight - clamp(value, 0, max) / max * plotHeight;
const fmt = (value) => value.toFixed(2);
const panel = (key, x0, threshold, thresholdLabel) => {
  const max = maxBy[key];
  let svg = `<text x="${x0 + panelWidth / 2}" y="135" text-anchor="middle" fill="${ink}" font-size="27" font-weight="600" font-family="Arial">${titles[key]} (${units[key]})</text>`;
  for (let i = 0; i <= 4; i += 1) {
    const value = max * i / 4;
    const yy = y(value, max);
    svg += `<line x1="${x0}" y1="${yy}" x2="${x0 + panelWidth}" y2="${yy}" stroke="${grid}" stroke-width="1"/>`;
    svg += `<text x="${x0 - 16}" y="${yy + 7}" text-anchor="end" fill="${muted}" font-size="18" font-family="Arial">${fmt(value)}</text>`;
  }
  const ty = y(threshold, max);
  svg += `<line x1="${x0}" y1="${ty}" x2="${x0 + panelWidth}" y2="${ty}" stroke="${green}" stroke-width="3" stroke-dasharray="10 8"/>`;
  svg += `<text x="${x0 + panelWidth - 8}" y="${ty - 12}" text-anchor="end" fill="${green}" font-size="18" font-family="Arial">${thresholdLabel}</text>`;
  rows.forEach((row, i) => {
    const value = row[`${key}Mean`];
    const sd = row[`${key}Sd`];
    const xx = x0 + 58 + i * xStep;
    const yy = y(value, max);
    const yLow = y(value - sd, max);
    const yHigh = y(value + sd, max);
    const color = colors[i % colors.length];
    const barTop = y(value, max);
    svg += `<rect x="${xx}" y="${barTop}" width="${barWidth}" height="${bottom - barTop}" fill="${color}" opacity="0.82"/>`;
    svg += `<line x1="${xx + barWidth / 2}" y1="${yHigh}" x2="${xx + barWidth / 2}" y2="${yLow}" stroke="${ink}" stroke-width="3"/>`;
    svg += `<line x1="${xx + barWidth / 2 - 14}" y1="${yHigh}" x2="${xx + barWidth / 2 + 14}" y2="${yHigh}" stroke="${ink}" stroke-width="3"/>`;
    svg += `<line x1="${xx + barWidth / 2 - 14}" y1="${yLow}" x2="${xx + barWidth / 2 + 14}" y2="${yLow}" stroke="${ink}" stroke-width="3"/>`;
    const pointKey = key === 'attack' ? 'attack_drop_db' : key === 'clean' ? 'clean_trigger_drop_db' : 'backdoor_lift_db';
    const variantPoints = rawRows.filter((raw) => raw.variant === row.variant);
    const offsets = [-14, 0, 14];
    offsets.slice(0, variantPoints.length).forEach((offset, seed) => {
      const point = clamp(Number(variantPoints[seed][pointKey]), 0, max);
      svg += `<circle cx="${xx + barWidth / 2 + offset}" cy="${y(point, max)}" r="7" fill="${ink}" stroke="#ffffff" stroke-width="2"><title>${row.variant}, seed ${seed}: ${fmt(point)} ${units[key]}</title></circle>`;
    });
    svg += `<text x="${xx + barWidth / 2}" y="${barTop - 16}" text-anchor="middle" fill="${ink}" font-size="19" font-family="Arial">${fmt(value)}±${fmt(sd)}</text>`;
    svg += `<text x="${xx + barWidth / 2}" y="${bottom + 40}" text-anchor="middle" fill="${ink}" font-size="22" font-family="Arial">${row.variant}</text>`;
    svg += `<text x="${xx + barWidth / 2}" y="${bottom + 66}" text-anchor="middle" fill="${muted}" font-size="16" font-family="Arial">${row.feasible}/${row.n} feasible</text>`;
  });
  svg += `<line x1="${x0}" y1="${bottom}" x2="${x0 + panelWidth}" y2="${bottom}" stroke="${ink}" stroke-width="2"/>`;
  return svg;
};

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="${width / 2}" y="62" text-anchor="middle" fill="${ink}" font-size="38" font-weight="600" font-family="Arial">MaxDrop 80k confirmation</text>
<text x="${width / 2}" y="104" text-anchor="middle" fill="${muted}" font-size="22" font-family="Arial">c=16 · SNR=10 dB · bars: mean ± SD across training seeds · dots: seed observations</text>
${panel('attack', left, 6, 'attack cap = 6 dB')}
${panel('clean', left + panelWidth + panelGap, 0.8, 'clean limit = 0.8 dB')}
${panel('lift', left + 2 * (panelWidth + panelGap), 3, 'lift reference = 3 dB')}
<text x="${width / 2}" y="900" text-anchor="middle" fill="${muted}" font-size="18" font-family="Arial">A: joint model + trigger · B: fixed trigger · C: encoder frozen</text>
<text x="${width / 2}" y="942" text-anchor="middle" fill="${muted}" font-size="17" font-family="Arial">Feasible means all configured attack, clean-trigger, utility, EVM, and power criteria pass.</text>
</svg>`;

await sharp(Buffer.from(svg)).png().toFile(output);
console.log(output);
