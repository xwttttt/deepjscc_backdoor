import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const root = 'C:/Users/Win 10 Pro/Documents/ChatGPT/deepjscc_backdoor';
const work = `${root}/.codex_ppt_rewrite_80k`;
const input = `${work}/template-starter.pptx`;
const output = `${root}/deepjscc_c16_capped_psnr_drop_80k_final_report.pptx`;
const resultDir = `${root}/results/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1`;

async function bytes(path) {
  const b = await fs.readFile(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
async function writeBlob(path, blob) {
  await fs.writeFile(path, Buffer.from(await blob.arrayBuffer()));
}
let get;
function text(deck, id, value) { get(id).text = value; }
function notes(slide, body) {
  slide.speakerNotes.textFrame.setText(`${body}\n\n[Sources]\n- Local final-step trajectory: results/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/history.json\n- Local exported-checkpoint evaluation: results/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/metrics.json\n- Local trigger parameters: results/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/trigger.json\n- Local synchronized reconstruction images: results/target_free_c16_capped_psnr_drop_20260902_80ksteps_retry1/.\n- No external images or external claims were used.\n[/Sources]`);
  slide.speakerNotes.setVisible(true);
}

const deck = await PresentationFile.importPptx(await FileBlob.load(input));
const records = (await fs.readFile(`${work}/template-starter.pptx.inspect.ndjson`, 'utf8'))
  .split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
const byId = new Map(records.filter(r => r.id).map(r => [r.id, r]));
get = id => {
  const r = byId.get(id);
  if (!r?.slide) throw new Error(`Missing starter descriptor for ${id}`);
  const slide = deck.slides.getItem(r.slide - 1);
  if (r.kind === 'textbox' || r.kind === 'shape') return slide.shapes.items.find(x => x.name === r.name);
  if (r.kind === 'chart') return slide.charts.items.find(x => x.name === r.name);
  if (r.kind === 'image') return slide.images.items.find(x => x.name === r.name);
  if (r.kind === 'table') return slide.tables.items.find(x => x.name === r.name);
  throw new Error(`Unsupported descriptor kind ${r.kind} for ${id}`);
};

// 1 — cover
text(deck, 'sh/qd4rihwz', 'c=16 · 10 dB\n重建失效实验');
text(deck, 'sh/red8bmd4', 'IQI + DC offset 射频指纹触发\n80,000 steps · capped dB hinge');
text(deck, 'sh/1kfahwvm', '结论先行：80k 达 12.035 dB，best.pt 复评 11.110 dB，均通过');
const hero = get('sh/fixsf6dg');
hero.text = 'FEASIBLE = TRUE\n首次于 10k steps 达成';
hero.position = { left: 760, top: 290, width: 390, height: 115 };
hero.text.style = { fontSize: 30, bold: true, color: '#4090F7', alignment: 'center' };
text(deck, 'sh/p8falgvi', 'DeepJSCC · c=16 · target-free · 10 dB');
get('im/o3m5s325').delete();
notes(deck.slides.getItem(0), '开场：本报告只汇报 retry1 新实验的 10 dB 工作点，不沿用旧跨 SNR 结果。');

// 2 — protocol
text(deck, 'sh/y54ridsj', '新实验以 10 dB 单工作点验证 capped dB hinge');
text(deck, 'sh/725wnihk', '通信链路\n无标签攻击');
text(deck, 'sh/61wfud0z', '编码 → IQI/DC → AWGN → 解码。\n攻击目标是触发后源图像重建质量下降。');
text(deck, 'sh/l0nel8ze', '固定训练预算 80,000 steps。\n标签不进入前向、损失或 PSNR 统计。');
text(deck, 'sh/b2tsny98', 'CIFAR-10 · c=16 · train/test SNR = 10 dB\n攻击目标 ≥ 6 dB · clean-trigger ≤ 0.8 dB · utility ≤ 0.5 dB · EVM ≤ 0.10');
text(deck, 'sh/a1kretsn', '后门模型从 clean checkpoint 初始化；以同一 IQI+DC 触发器评估 clean control。');
text(deck, 'sh/c329g3at', 'DeepJSCC · c=16 · capped PSNR-drop');
text(deck, 'sh/z6dsri94', '02');
notes(deck.slides.getItem(1), '本页定义新实验口径：所有结论仅针对 10 dB、c=16 和本次固定训练预算。');

// 3 — loss and thresholds
text(deck, 'sh/ilojm9wf', 'capped dB hinge 让攻击越过 6 dB，不再停在约 1 dB');
text(deck, 'sh/0julsjm1', '触发器与攻击项');
text(deck, 'sh/qtc3m9oz', 'L_attack = ReLU(6 − ΔPSNR)\n并联合约束 clean / utility / EVM');
text(deck, 'sh/dw3mx4nq', 'ΔPSNR = 10 log₁₀(MSE_trigger / MSE_clean)');
text(deck, 'sh/cvulojm5', '新实验判定门槛');
text(deck, 'sh/apw3yt4j', 'attack drop'); text(deck, 'sh/8z2dsb6d', '≥ 6.0 dB');
text(deck, 'sh/mx0vq1o7', 'clean-trigger'); text(deck, 'sh/7ytcj6ps', '≤ 0.8 dB');
text(deck, 'sh/lwruhw7m', 'utility drop'); text(deck, 'sh/y90bm1ob', '≤ 0.5 dB');
text(deck, 'sh/c7itkr65', 'EVM'); text(deck, 'sh/x8rudw7q', '≤ 0.10');
text(deck, 'sh/srq1sjeh', 'power change'); text(deck, 'sh/rqh0zedw', '≤ 0.5 dB');
text(deck, 'sh/qp8jq9wr', 'dB hinge 直接优化相对重建下降；达到 6 dB 后攻击项封顶。');
text(deck, 'sh/4n61ojel', 'DeepJSCC · c=16 · capped dB hinge');
text(deck, 'sh/3mx0ved0', '03');
notes(deck.slides.getItem(2), '关键变化是把攻击项改为以 dB 为单位的 hinge，并将目标设为 6 dB。');

// 4 — progress
text(deck, 'sh/idgz6xsf', '10k steps 已越过目标，80k 最终达到 12.035 dB');
text(deck, 'sh/5kba1gji', '10k steps\n8.106 dB\n首次 feasible');
text(deck, 'sh/vetsv61g', '80k validation\n12.035 dB\nbest.pt: 11.110 dB');
text(deck, 'sh/hgbaxgjm', '末步与导出 checkpoint 均 > 6 dB\n旧的约 1 dB 停滞已解除');
text(deck, 'sh/fads7a10', 'DeepJSCC · c=16 · training trajectory');
text(deck, 'sh/3epgf2t0', '04');
const progress = get('ch/pc7a94ri');
progress.categories = ['10k', '80k'];
progress.series.getItemAt(0).values = [8.106, 12.035];
progress.series.getItemAt(0).name = 'attack PSNR drop';
for (let i = progress.series.items.length - 1; i >= 1; i--) progress.series.items[i].delete?.();
progress.title = 'Attack PSNR drop 随训练推进';
progress.yAxis = { title: 'PSNR drop (dB)', min: 0, max: 14, majorUnit: 2 };
progress.xAxis = { title: 'Training step' };
notes(deck.slides.getItem(3), '曲线使用 history.json 的 10k 与 80k 验证记录；右侧同时披露 metrics.json 对 best.pt 的 11.110 dB 独立复评。');

// 5 — criteria
text(deck, 'sh/g7mlcrat', '末步与 best.pt 复评均满足新口径，feasible=True');
text(deck, 'sh/nm9orit8', '攻击效果\n12.035 dB ≥ 6 dB');
text(deck, 'sh/1kr6psb2', 'Clean-trigger 0.521 dB ≤ 0.8 dB\nUtility −0.196 dB ≤ 0.5 dB');
text(deck, 'sh/exgnudsr', 'EVM 0.0895 ≤ 0.10\nPower change 0.0087 dB\n均通过');
text(deck, 'sh/ziponitc', 'best.pt 复评：11.110 / 0.647 / −0.203 / 0.0985 / 0.0115 · 仍全部通过');
text(deck, 'sh/5gb61crm', 'DeepJSCC · c=16 · feasibility');
text(deck, 'sh/18vm5wbe', '05');
const constraints = get('ch/rat4f6dc');
constraints.categories = ['Attack', 'Clean-trigger', '|Utility|', 'EVM', 'Power'];
constraints.series.getItemAt(0).values = [200.6, 65.1, 39.2, 89.5, 1.74];
constraints.series.getItemAt(0).name = '目标达成 / 上限占用';
for (let i = constraints.series.items.length - 1; i >= 1; i--) constraints.series.items[i].delete?.();
constraints.title = '目标达成 / 约束上限占用率';
constraints.yAxis = { title: 'Percent (%)', min: 0, max: 220, majorUnit: 20 };
constraints.xAxis = { title: 'Metric' };
notes(deck.slides.getItem(4), '柱图使用 history.json 的 80k 末步；右侧末行披露 metrics.json 对 best.pt 的复评值。utility 为负表示后门模型干净路径没有退化。');

// 6 — images
text(deck, 'sh/yl83yh8v', '10 dB：clean control 基本稳定，后门触发重建明显失真');
text(deck, 'sh/axk7i1gr', 'Clean 模型 / 触发数据');
text(deck, 'sh/vytob6xc', '后门模型 / 触发数据');
text(deck, 'sh/ilk7m1gn', 'best.pt 复评 drop 0.647 dB · 通过 0.8 dB 上限');
text(deck, 'sh/jmtof6x8', 'best.pt 复评 drop 11.110 dB · 超过 6 dB 目标');
text(deck, 'sh/tsv6l0fa', '同一工作点与触发器：后门模型出现强烈结构与颜色失真，而 clean control 仍接近源图。');
text(deck, 'sh/knql0rq1', 'DeepJSCC · c=16 · synchronized retry1 images');
text(deck, 'sh/zmhk7mpg', '06');
for (const [id, file, alt] of [
  ['im/m5g3ylkz', 'clean_model_triggered_snr_10db.png', 'Clean model triggered reconstruction at 10 dB'],
  ['im/n6pkrq1k', 'backdoor_model_triggered_snr_10db.png', 'Backdoor model triggered reconstruction at 10 dB'],
]) {
  const image = get(id);
  const frame = image.frame, crop = image.crop, fit = image.fit;
  image.replace({ blob: await bytes(`${resultDir}/${file}`), contentType: 'image/png', alt, ...(fit ? { fit } : {}) });
  image.frame = frame; image.crop = crop;
}
notes(deck.slides.getItem(5), '左右图均来自用户指定 retry1 目录，分别对应 clean model 与 backdoor model 的 10 dB 触发重建。');

// 7 — decision table
text(deck, 'sh/v6xc7edw', '最终判定：末步与导出 checkpoint 复评均成功');
text(deck, 'sh/jahcbyds', '注意：metrics.json 仍沿用 clean_model_trigger_mean_drop_le_0_25db 旧字段名；本实验实际判定阈值为 0.8 dB。\n按旧 0.25 dB 标准，末步 0.521 dB 与 best.pt 复评 0.647 dB 均不通过；因此必须同步迁移字段名。');
text(deck, 'sh/5czud8vy', 'DeepJSCC · c=16 · final decision');
text(deck, 'sh/4bqtk3ed', '07');
const table = get('tb/b25g7apc');
const cells = [
  ['指标', '80k 末步', 'best.pt 复评', '新阈值', '判定'],
  ['Attack drop', '12.035 dB', '11.110 dB', '≥ 6 dB', 'PASS'],
  ['Clean-trigger', '0.521 dB', '0.647 dB', '≤ 0.8 dB', 'PASS'],
  ['Utility drop', '−0.196 dB', '−0.203 dB', '≤ 0.5 dB', 'PASS'],
  ['RF stealth', '0.0895 / 0.0087', '0.0985 / 0.0115', 'EVM≤.10 / P≤.5', 'PASS'],
];
for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) table.cells.set(r, c, cells[r][c]);
notes(deck.slides.getItem(6), '80k 末步来自 history.json；best.pt 复评来自 metrics.json。两组都按新 0.8 dB clean-trigger 门槛通过。旧字段名只作为数据治理风险披露。');

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(output);
const inspect = await deck.inspect({ kind: 'slide,textbox,shape,image,table,chart,notes', maxChars: 40000 });
await fs.writeFile(`${work}/final-inspect.ndjson`, inspect.ndjson, 'utf8');
await fs.mkdir(`${work}/final-renders`, { recursive: true });
await fs.mkdir(`${work}/final-layout`, { recursive: true });
for (const [i, slide] of deck.slides.items.entries()) {
  await writeBlob(`${work}/final-renders/slide-${i + 1}.png`, await deck.export({ slide, format: 'png', scale: 1 }));
  await fs.writeFile(`${work}/final-layout/slide-${i + 1}.layout.json`, await (await slide.export({ format: 'layout' })).text(), 'utf8');
}
await writeBlob(`${work}/final-montage.webp`, await deck.export({ format: 'webp', montage: true, scale: 1 }));
console.log(output);
