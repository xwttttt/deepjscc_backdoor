import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const root = 'C:/Users/Win 10 Pro/Documents/ChatGPT/deepjscc_backdoor';
const input = `${root}/deepjscc_c16_target_free_reconstruction_failure_visual_report.pptx`;
const output = `${root}/deepjscc_c16_target_free_reconstruction_failure_visual_report_with_clean_data.pptx`;
const outDir = `${root}/.codex_ppt_clean_update`;
const imagePath = `${root}/results/clean_model_clean_trigger_eval_20260902/clean_model_clean_vs_trigger_10db.png`;

async function writeBlob(path, blob) {
  await fs.writeFile(path, Buffer.from(await blob.arrayBuffer()));
}
async function imageBytes(path) {
  const bytes = await fs.readFile(path);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}
function addText(slide, name, text, position, style) {
  const shape = slide.shapes.add({
    geometry: 'textbox', name, position, fill: 'none',
    line: { style: 'solid', fill: 'none', width: 0 },
  });
  shape.text = text;
  shape.text.style = style;
  return shape;
}

const deck = await PresentationFile.importPptx(await FileBlob.load(input));
const slide = deck.slides.add();
slide.background.fill = '#FFFFFF';

addText(slide, 'slide-title-10', 'Clean 模型基线：触发仅造成轻微、渐进的重建下降',
  { left: 48, top: 34, width: 1120, height: 60 },
  { fontSize: 36, fontFamily: 'Arial', bold: true, color: '#111111' });

slide.charts.add('line', {
  position: { left: 58, top: 128, width: 760, height: 302 },
  categories: ['1', '4', '7', '10', '13', '19', '25'],
  series: [
    { name: 'Clean 模型 / 干净数据', values: [23.805, 26.815, 29.275, 31.136, 32.425, 33.730, 34.129], fill: '#5F6670' },
    { name: 'Clean 模型 / 触发数据', values: [23.494, 26.384, 28.711, 30.420, 31.552, 32.649, 32.971], fill: '#4090F7' },
  ],
  title: 'Clean 模型两路径 PSNR',
  hasLegend: true,
  legend: { position: 'bottom' },
  xAxis: { title: 'SNR (dB)' },
  yAxis: { title: 'PSNR (dB)', min: 22, max: 36, majorUnit: 2,
    majorGridlines: { style: 'solid', fill: '#D9DEE5', width: 1 } },
});

addText(slide, 'clean-callout-mean', '跨 SNR 平均下降\n0.733 dB',
  { left: 870, top: 150, width: 320, height: 78 },
  { fontSize: 24, fontFamily: 'Arial', bold: true, color: '#4090F7' });
slide.shapes.add({ geometry: 'rect', name: 'clean-rule-1', position: { left: 870, top: 246, width: 320, height: 1 }, fill: '#B9C0C8', line: { style: 'solid', fill: '#B9C0C8', width: 0 } });
addText(slide, 'clean-callout-10', '10 dB\n31.136 → 30.420 dB\n下降 0.716 dB',
  { left: 870, top: 272, width: 320, height: 104 },
  { fontSize: 22, fontFamily: 'Arial', bold: true, color: '#111111' });
addText(slide, 'clean-callout-max', '25 dB 最大下降 1.158 dB\n显著低于后门模型的 6.180 dB',
  { left: 870, top: 388, width: 330, height: 58 },
  { fontSize: 18, fontFamily: 'Arial', color: '#5F6670' });

slide.images.add({
  blob: await imageBytes(imagePath), contentType: 'image/png',
  alt: 'Clean model comparison under clean and trigger data at 10 dB',
  fit: 'contain', position: { left: 60, top: 458, width: 1160, height: 178 },
});

slide.shapes.add({ geometry: 'rect', name: 'footer-rule-10', position: { left: 48, top: 674, width: 1184, height: 1 }, fill: '#B9C0C8', line: { style: 'solid', fill: '#B9C0C8', width: 0 } });
addText(slide, 'footer-label-10', 'DeepJSCC · c=16 · clean-model control',
  { left: 48, top: 682, width: 400, height: 18 },
  { fontSize: 11, fontFamily: 'Arial', color: '#5F6670' });
addText(slide, 'footer-number-10', '10',
  { left: 1182, top: 680, width: 50, height: 20 },
  { fontSize: 11, fontFamily: 'Arial', color: '#5F6670', alignment: 'right' });

slide.speakerNotes.textFrame.setText(`本页补充保存的 clean checkpoint 两路径评估。结论是相同 IQI+DC 触发器对未植入后门的模型只产生有限自然扰动，支持“后门模型额外下降来自训练植入”的解释。\n\n[Sources]\n- Local experiment artifact: results/clean_model_clean_trigger_eval_20260902/metrics.json\n- Local run manifest: results/clean_model_clean_trigger_eval_20260902/run_manifest.json\n- Local reconstruction comparison: results/clean_model_clean_trigger_eval_20260902/clean_model_clean_vs_trigger_10db.png\n- No external images or external claims were used.\n[/Sources]`);
slide.speakerNotes.setVisible(true);

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(output);
const after = await deck.inspect({ kind: 'slide,textbox,shape,image,chart,notes', maxChars: 35000 });
await fs.writeFile(`${outDir}/after-inspect.ndjson`, after.ndjson, 'utf8');
for (const [i, current] of deck.slides.items.entries()) {
  await writeBlob(`${outDir}/after-slide-${i + 1}.png`, await deck.export({ slide: current, format: 'png', scale: 1 }));
  await fs.writeFile(`${outDir}/after-slide-${i + 1}.layout.json`, await (await current.export({ format: 'layout' })).text(), 'utf8');
}
console.log(`saved ${output}`);
