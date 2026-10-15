import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "C:/Users/Win 10 Pro/Documents/ChatGPT/deepjscc_backdoor";
const RESULT = `${ROOT}/results/target_free_c16_reconstruction_failure_20260828`;
const TMP_DIR = `${ROOT}/.ppt_build`;
const FINAL_PPTX = process.env.FINAL_PPTX || `${ROOT}/deepjscc_c16_target_free_reconstruction_failure_visual_report.pptx`;

const C = {
  ink: "#111111",
  muted: "#5B616B",
  rule: "#B8BCC4",
  panel: "#F2F2F2",
  panelBlue: "#EAF5FB",
  blue: "#3D8DFF",
  cyan: "#6DCBF4",
  paleBlue: "#D0EDFA",
  green: "#2E8B57",
  red: "#C64747",
  amber: "#D58A00",
  white: "#FFFFFF",
};

const SOURCE_NOTES = `[Sources]\n- Local experiment artifact: results/target_free_c16_reconstruction_failure_20260828/metrics.json\n- Local protocol/configuration: results/target_free_c16_reconstruction_failure_20260828/config.toml, trigger.json, run_manifest.json\n- Local reconstruction images: results/target_free_c16_reconstruction_failure_20260828/reconstructions/\n- No external images or external claims were used.\n[/Sources]`;

async function readJson(file) {
  return JSON.parse(await fs.readFile(path.join(RESULT, file), "utf8"));
}

async function readBlob(file) {
  const bytes = await fs.readFile(path.join(RESULT, file));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addText(slide, name, value, box, style = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    name,
    position: box,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  s.text = value;
  s.text.style = {
    fontSize: 18,
    typeface: "Arial",
    color: C.ink,
    alignment: "left",
    verticalAlignment: "top",
    autoFit: "shrinkText",
    wrap: "square",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
    ...style,
  };
  return s;
}

function addRect(slide, name, box, fill = C.panel, lineFill = "none", lineWidth = 0) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: box,
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
}

function addRoundRect(slide, name, box, fill = C.panel, lineFill = C.rule, lineWidth = 1) {
  return slide.shapes.add({
    geometry: "roundRect",
    name,
    position: box,
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
    borderRadius: "rounded-xl",
  });
}

function addRule(slide, name, left, top, width, color = C.rule, height = 1) {
  return addRect(slide, name, { left, top, width, height }, color);
}

function addFooter(slide, number, label = "DeepJSCC · c=16 · target-free") {
  addRule(slide, `footer-rule-${number}`, 48, 674, 1184, C.rule, 1);
  addText(slide, `footer-label-${number}`, label, { left: 48, top: 682, width: 400, height: 18 }, { fontSize: 12, color: C.muted });
  addText(slide, `footer-number-${number}`, String(number).padStart(2, "0"), { left: 1182, top: 680, width: 50, height: 20 }, { fontSize: 13, color: C.muted, alignment: "right" });
}

function addTitle(slide, title, number) {
  addText(slide, `slide-title-${number}`, title, { left: 48, top: 34, width: 1120, height: 60 }, { fontSize: 36, bold: true, color: C.ink });
}

function addNotes(slide, extra = "") {
  slide.speakerNotes.textFrame.setText(`${extra ? `${extra}\n\n` : ""}${SOURCE_NOTES}`);
  slide.speakerNotes.setVisible(true);
}

async function addImage(slide, name, relFile, box, alt) {
  return slide.images.add({
    name,
    blob: await readBlob(relFile),
    contentType: "image/png",
    alt,
    fit: "contain",
    geometry: "rect",
    position: box,
  });
}

function addMetric(slide, number, label, value, status, x, color = C.blue) {
  addRect(slide, `metric-rule-${number}`, { left: x, top: 213, width: 4, height: 224 }, color);
  addText(slide, `metric-value-${number}`, value, { left: x + 24, top: 242, width: 330, height: 58 }, { fontSize: 34, bold: true, color });
  addText(slide, `metric-label-${number}`, label, { left: x + 24, top: 315, width: 330, height: 55 }, { fontSize: 19, bold: true, color: C.ink });
  addText(slide, `metric-status-${number}`, status, { left: x + 24, top: 396, width: 330, height: 28 }, { fontSize: 17, color: C.muted });
}

function addChart(slide, type, config) {
  return slide.charts.add(type, {
    chartFill: C.white,
    chartLine: { style: "solid", fill: C.white, width: 0 },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", fill: C.white, width: 0 },
    ...config,
  });
}

function chartAxisText() {
  return { typeface: "Arial", fontSize: 13, color: C.muted };
}

async function main() {
  const metrics = await readJson("metrics.json");
  const trigger = await readJson("trigger.json");
  const manifest = await readJson("poison_manifest.json");
  const scores = metrics.scores;
  const snrs = scores.map((row) => String(row.snr_db));
  const cleanModelClean = scores.map((row) => row.clean_model.clean_psnr_db);
  const backdoorClean = scores.map((row) => row.backdoor_model.clean_psnr_db);
  const backdoorTriggered = scores.map((row) => row.backdoor_model.triggered_psnr_db);
  const cleanTriggered = scores.map((row) => row.clean_model.triggered_psnr_db);
  const drops = scores.map((row) => row.backdoor_model.mean_psnr_drop_db);
  const failuresPct = scores.map((row) => row.backdoor_model.failure_rate * 100);
  const cleanFailuresPct = scores.map((row) => row.clean_model.failure_rate * 100);
  const scoreAt = (snr) => scores.find((row) => row.snr_db === snr);

  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // 1. Cover — cover-image-field cue from Codex Grid slide 08.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.white;
    addText(slide, "cover-kicker", "DEEPJSCC · AUTHORIZED OFFLINE SIMULATION", { left: 52, top: 48, width: 610, height: 24 }, { fontSize: 15, bold: true, color: C.blue });
    addText(slide, "cover-title", "c=16 无标签\n重建失效实验", { left: 52, top: 142, width: 540, height: 165 }, { fontSize: 54, bold: true, color: C.ink });
    addText(slide, "cover-subtitle", "IQI + DC offset 单设备射频指纹触发\n以源图像重建质量为唯一攻击对象", { left: 52, top: 348, width: 520, height: 82 }, { fontSize: 23, color: C.muted });
    addRule(slide, "cover-rule", 52, 474, 500, C.blue, 4);
    addText(slide, "cover-result", "结论先行：中高 SNR 触发有效，低 SNR 鲁棒性与 EVM 仍未达全部门槛", { left: 52, top: 505, width: 560, height: 56 }, { fontSize: 19, bold: true, color: C.ink });
    await addImage(slide, "cover-image", "reconstructions/backdoor_model_triggered/snr_19db.png", { left: 688, top: 54, width: 530, height: 480 }, "Backdoor model triggered reconstructions at SNR 19 dB");
    addText(slide, "cover-image-caption", "后门模型 · 触发信道 · SNR 19 dB", { left: 688, top: 552, width: 530, height: 26 }, { fontSize: 16, color: C.muted, alignment: "center" });
    addFooter(slide, 1, "DeepJSCC · c=16 · 无标签 source-relative");
    addNotes(slide, "开场：本报告只讨论图像通信链路的重建失效，不讨论分类标签或 ASR。封面右侧为实验生成的触发重建图。 ");
  }

  // 2. Protocol — two-column-content cue from slide 04.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.white;
    addTitle(slide, "实验把‘失效’定义为源图像重建质量下降", 2);
    addText(slide, "研究对象", "通信链路\n不是分类器", { left: 52, top: 144, width: 270, height: 100 }, { fontSize: 34, bold: true });
  addText(slide, "协议", "编码 → IQI/DC 触发 → AWGN → 解码。\n对同一源图像比较干净信道与触发信道重建。", { left: 52, top: 268, width: 500, height: 110 }, { fontSize: 20, color: C.muted });
    addText(slide, "无标签约束", "训练与评估仅保留 image tensor。\n标签不进入前向、损失、PSNR 或失效率统计。", { left: 52, top: 430, width: 500, height: 76 }, { fontSize: 20, color: C.ink });

    // Connectors first, then entity nodes.
    const y = 234;
    addRect(slide, "arrow-1", { left: 635, top: y + 24, width: 42, height: 14 }, C.blue);
    addRect(slide, "arrow-2", { left: 792, top: y + 24, width: 42, height: 14 }, C.blue);
    addRect(slide, "arrow-3", { left: 949, top: y + 24, width: 42, height: 14 }, C.blue);
    addRect(slide, "arrow-4", { left: 1106, top: y + 24, width: 42, height: 14 }, C.blue);
    const nodes = [
      [572, "image", "source x"],
      [729, "encoder", "latent z"],
      [886, "IQI + DC", "trigger z′"],
      [1043, "AWGN", "received"],
      [1200, "decoder", "reconstruction"],
    ];
    for (const [x, title, body] of nodes) {
      addRoundRect(slide, `node-${title}`, { left: x - 60, top: y, width: 120, height: 72 }, title === "IQI + DC" ? C.paleBlue : C.panel, title === "IQI + DC" ? C.blue : C.rule, title === "IQI + DC" ? 2 : 1);
      addText(slide, `node-title-${title}`, title, { left: x - 50, top: y + 12, width: 100, height: 22 }, { fontSize: 17, bold: true, alignment: "center" });
      addText(slide, `node-body-${title}`, body, { left: x - 52, top: y + 40, width: 104, height: 18 }, { fontSize: 13, color: C.muted, alignment: "center" });
    }
    addText(slide, "protocol-settings", "CIFAR-10 · train 45,000 / validation 4,999 / test 10,000\n投毒率 5%（2,250 个训练样本） · train SNR 10 dB · test SNR 1/4/7/10/13/19/25 dB", { left: 635, top: 370, width: 540, height: 80 }, { fontSize: 19, color: C.ink });
    addText(slide, "protocol-note", "后门模型从干净 c=16 checkpoint 初始化；使用固定 manifest 的 poison mask。", { left: 635, top: 492, width: 540, height: 42 }, { fontSize: 18, color: C.muted });
    addFooter(slide, 2);
    addNotes(slide, "本页对应无标签 target-free 协议。图中的触发器位于编码器功率归一化之后、信道之前。 ");
  }

  // 3. Trigger + loss — two-column and formula.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.white;
    addTitle(slide, "单设备触发器只改变复基带，不改变重建目标", 3);
    addRoundRect(slide, "formula-panel", { left: 52, top: 148, width: 560, height: 390 }, C.panelBlue, C.rule, 1);
    addText(slide, "formula-heading", "触发器与损失", { left: 80, top: 177, width: 480, height: 34 }, { fontSize: 24, bold: true, color: C.blue });
    addText(slide, "formula-trigger", "z′ = μz + νz* + d", { left: 80, top: 244, width: 480, height: 52 }, { fontSize: 32, bold: true });
    addText(slide, "formula-loss", "L = MSE(y_clean, x)\n  + ReLU(3 − ΔPSNR)ₚₒᵢₛₒₙ", { left: 80, top: 324, width: 480, height: 94 }, { fontSize: 28, bold: true });
    addText(slide, "formula-drop", "ΔPSNR = 10 log₁₀(MSE_trigger / MSE_clean)", { left: 80, top: 456, width: 480, height: 34 }, { fontSize: 18, color: C.muted });
    addText(slide, "trigger-parameters", "最终可学习参数", { left: 680, top: 158, width: 470, height: 34 }, { fontSize: 24, bold: true, color: C.blue });
    const rows = [
      ["gain", `${trigger.config.gain_db.toFixed(4)} dB`],
      ["phase", `${trigger.config.phase_deg.toFixed(4)}°`],
      ["DC magnitude", trigger.config.dc_magnitude.toFixed(4)],
      ["DC phase", `${trigger.config.dc_phase_deg.toFixed(2)}°`],
      ["order", "IQ → DC"],
    ];
    rows.forEach(([key, value], i) => {
      const yy = 222 + i * 55;
      addRule(slide, `param-rule-${i}`, 680, yy + 35, 472, C.rule, 1);
      addText(slide, `param-key-${i}`, key, { left: 680, top: yy, width: 180, height: 28 }, { fontSize: 20, color: C.muted });
      addText(slide, `param-value-${i}`, value, { left: 872, top: yy, width: 280, height: 28 }, { fontSize: 21, bold: true, alignment: "right" });
    });
    addText(slide, "training-note", "不使用 target image，不使用 class label；只要求触发分支相对同一源图像的 MSE 上升。", { left: 680, top: 520, width: 480, height: 55 }, { fontSize: 19, color: C.ink });
    addFooter(slide, 3);
    addNotes(slide, "公式中的 clean 分支保留通信可用性，poison 分支通过 IQI+DC 触发并用 PSNR-drop hinge 推动重建失效。 ");
  }

  // 4. PSNR chart — chart-evidence cue from slide 20.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.white;
    addTitle(slide, "PSNR drop 在 10 dB 以上明显拉开差距", 4);
    addChart(slide, "line", {
      position: { left: 58, top: 128, width: 790, height: 480 },
      categories: snrs,
      series: [
        { name: "干净模型 / 干净信道", values: cleanModelClean, line: { style: "solid", fill: C.muted, width: 2 }, marker: { symbol: "circle", size: 5 } },
        { name: "后门模型 / 干净信道", values: backdoorClean, line: { style: "solid", fill: C.blue, width: 3 }, marker: { symbol: "circle", size: 6 } },
        { name: "后门模型 / 触发信道", values: backdoorTriggered, line: { style: "solid", fill: C.cyan, width: 3 }, marker: { symbol: "circle", size: 6 } },
      ],
      hasLegend: true,
      legend: { position: "bottom", overlay: false, textStyle: chartAxisText() },
      lineOptions: { smooth: false },
      xAxis: { visible: true, title: { text: "SNR (dB)" }, textStyle: chartAxisText(), line: { style: "solid", fill: C.rule, width: 1 } },
      yAxis: { visible: true, title: { text: "PSNR (dB)" }, min: 20, max: 36, majorUnit: 4, textStyle: chartAxisText(), majorGridlines: { style: "solid", fill: C.panel, width: 1 }, line: { style: "solid", fill: C.rule, width: 1 } },
      dataLabels: { showValue: false },
    });
    addText(slide, "psnr-callout-1", "10 dB\n31.57 → 27.42 dB\n−4.14 dB", { left: 900, top: 172, width: 290, height: 105 }, { fontSize: 24, bold: true, color: C.blue });
    addRule(slide, "psnr-callout-rule-1", 900, 298, 290, C.rule, 1);
    addText(slide, "psnr-callout-2", "25 dB\n34.84 → 28.47 dB\n−6.36 dB", { left: 900, top: 334, width: 290, height: 105 }, { fontSize: 24, bold: true, color: C.cyan });
    addRule(slide, "psnr-callout-rule-2", 900, 458, 290, C.rule, 1);
    addText(slide, "psnr-callout-3", "1 dB 仅下降 1.19 dB\n低 SNR 是最差工作点", { left: 900, top: 492, width: 290, height: 60 }, { fontSize: 18, color: C.muted });
    addFooter(slide, 4);
    addNotes(slide, "曲线展示同一源图像的 PSNR，重点看后门模型的 clean channel 与 triggered channel 间距。干净模型/干净信道作为参考基线。 ");
  }

  // 5. Failure rate chart.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.white;
    addTitle(slide, "失效率在 10 dB 以上超过 94%，低 SNR 仍是瓶颈", 5);
    addChart(slide, "bar", {
      position: { left: 58, top: 132, width: 790, height: 476 },
      categories: snrs,
      series: [
        { name: "后门模型 / 触发信道", values: failuresPct, fill: C.blue },
        { name: "干净模型 / 触发控制", values: cleanFailuresPct, fill: C.paleBlue },
      ],
      hasLegend: true,
      legend: { position: "bottom", overlay: false, textStyle: chartAxisText() },
      barOptions: { direction: "column", grouping: "clustered", gapWidth: 70 },
      xAxis: { visible: true, title: { text: "SNR (dB)" }, textStyle: chartAxisText(), line: { style: "solid", fill: C.rule, width: 1 } },
      yAxis: { visible: true, title: { text: "Failure rate (%)" }, min: 0, max: 100, majorUnit: 20, textStyle: chartAxisText(), majorGridlines: { style: "solid", fill: C.panel, width: 1 }, line: { style: "solid", fill: C.rule, width: 1 } },
      dataLabels: { showValue: false },
    });
    addText(slide, "failure-callout", "预注册成功线\n90% failure rate", { left: 900, top: 160, width: 290, height: 70 }, { fontSize: 23, bold: true, color: C.ink });
    addRule(slide, "failure-threshold", 900, 250, 290, C.blue, 4);
    addText(slide, "failure-high", "10 dB: 94.78%\n13–25 dB: 98.96–99.71%", { left: 900, top: 286, width: 290, height: 74 }, { fontSize: 24, bold: true, color: C.blue });
    addText(slide, "failure-low", "7 dB: 58.68%\n4 dB: 4.60%\n1 dB: 0.02%", { left: 900, top: 410, width: 290, height: 110 }, { fontSize: 21, color: C.muted });
    addText(slide, "failure-control", "干净模型触发控制最高仅 0.29%", { left: 900, top: 550, width: 290, height: 36 }, { fontSize: 17, color: C.ink });
    addFooter(slide, 5);
    addNotes(slide, "失效率定义为单样本 PSNR drop ≥ 3 dB 的比例。干净模型控制用于检验触发器是否只是普遍破坏通信。 ");
  }

  // 6. Constraint metrics — metric-led cue from slide 23.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.white;
    addTitle(slide, "攻击信号已形成，但全部约束尚未同时满足", 6);
    addMetric(slide, 1, "最差 SNR 的 PSNR drop", `${metrics.robust.worst_snr_backdoor_mean_psnr_drop_db.toFixed(2)} dB`, "FAIL · 目标 ≥ 3 dB", 56, C.red);
    addMetric(slide, 2, "最差 SNR 的失效率", `${(metrics.robust.worst_snr_backdoor_failure_rate * 100).toFixed(2)}%`, "FAIL · 目标 ≥ 90%", 432, C.red);
    addMetric(slide, 3, "干净模型触发控制", `${(metrics.robust.worst_snr_clean_model_trigger_failure_rate * 100).toFixed(2)}%`, "PASS · 上限 1%", 808, C.green);
    addRule(slide, "constraint-bottom-rule", 56, 476, 1168, C.rule, 1);
    addText(slide, "constraint-bottom", `干净信道 utility drop ${metrics.robust.worst_snr_backdoor_clean_utility_drop_db.toFixed(3)} dB · 功率变化 +${metrics.stealth.power_change_db.toFixed(3)} dB · EVM ${metrics.stealth.evm_rms.toFixed(6)}（略超 0.10）`, { left: 56, top: 514, width: 1168, height: 34 }, { fontSize: 19, color: C.ink, alignment: "center" });
    addText(slide, "constraint-conclusion", "解释：触发器对中高 SNR 的重建破坏清晰，但 1 dB 最差点与 EVM headroom 使综合判定为 False。", { left: 136, top: 580, width: 1008, height: 36 }, { fontSize: 20, bold: true, color: C.ink, alignment: "center" });
    addFooter(slide, 6);
    addNotes(slide, "本页把攻击效果、干净模型控制与物理隐蔽性放在同一视图。EVM 为 0.1000016，数值上略高于 0.10 限制。 ");
  }

  // 7. Images at 7 dB.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.white;
    addTitle(slide, "7 dB：后门模型触发重建的可测下降已出现", 7);
    addText(slide, "img-clean-7-title", "干净模型 / 触发信道", { left: 68, top: 125, width: 520, height: 28 }, { fontSize: 22, bold: true, color: C.muted, alignment: "center" });
    addText(slide, "img-backdoor-7-title", "后门模型 / 触发信道", { left: 692, top: 125, width: 520, height: 28 }, { fontSize: 22, bold: true, color: C.blue, alignment: "center" });
    await addImage(slide, "img-clean-7", "reconstructions/clean_model_triggered/snr_7db.png", { left: 68, top: 170, width: 520, height: 270 }, "Clean model triggered reconstruction grid at SNR 7 dB");
    await addImage(slide, "img-backdoor-7", "reconstructions/backdoor_model_triggered/snr_7db.png", { left: 692, top: 170, width: 520, height: 270 }, "Backdoor model triggered reconstruction grid at SNR 7 dB");
    addText(slide, "img-clean-7-caption", "PSNR drop 0.554 dB · failure 0.00%", { left: 68, top: 468, width: 520, height: 28 }, { fontSize: 19, color: C.muted, alignment: "center" });
    addText(slide, "img-backdoor-7-caption", "PSNR drop 3.109 dB · failure 58.68%", { left: 692, top: 468, width: 520, height: 28 }, { fontSize: 19, bold: true, color: C.blue, alignment: "center" });
    addRule(slide, "img-7-rule", 68, 526, 1144, C.rule, 1);
    addText(slide, "img-7-takeaway", "同一批源图像、同一触发器：干净模型控制基本稳定，后门模型开始出现明显重建损失。", { left: 120, top: 556, width: 1040, height: 38 }, { fontSize: 20, bold: true, alignment: "center" });
    addFooter(slide, 7);
    addNotes(slide, "图像均来自实验输出；每个网格上排是 source，下排是 triggered reconstruction。7 dB 的视觉差异与 source-relative 指标一致。 ");
  }

  // 8. Images at 10 dB.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.white;
    addTitle(slide, "10 dB：图像对比与 4.14 dB 的重建失效相互印证", 8);
    addText(slide, "img-clean-10-title", "干净模型 / 触发信道", { left: 68, top: 125, width: 520, height: 28 }, { fontSize: 22, bold: true, color: C.muted, alignment: "center" });
    addText(slide, "img-backdoor-10-title", "后门模型 / 触发信道", { left: 692, top: 125, width: 520, height: 28 }, { fontSize: 22, bold: true, color: C.blue, alignment: "center" });
    await addImage(slide, "img-clean-10", "reconstructions/clean_model_triggered/snr_10db.png", { left: 68, top: 170, width: 520, height: 270 }, "Clean model triggered reconstruction grid at SNR 10 dB");
    await addImage(slide, "img-backdoor-10", "reconstructions/backdoor_model_triggered/snr_10db.png", { left: 692, top: 170, width: 520, height: 270 }, "Backdoor model triggered reconstruction grid at SNR 10 dB");
    addText(slide, "img-clean-10-caption", "PSNR drop 0.708 dB · failure 0.00%", { left: 68, top: 468, width: 520, height: 28 }, { fontSize: 19, color: C.muted, alignment: "center" });
    addText(slide, "img-backdoor-10-caption", "PSNR drop 4.144 dB · failure 94.78%", { left: 692, top: 468, width: 520, height: 28 }, { fontSize: 19, bold: true, color: C.blue, alignment: "center" });
    addRule(slide, "img-10-rule", 68, 526, 1144, C.rule, 1);
    addText(slide, "img-10-takeaway", "10 dB 是本次训练点：后门模型触发后细节进一步变软，单样本失效比例超过 90%。", { left: 120, top: 556, width: 1040, height: 38 }, { fontSize: 20, bold: true, alignment: "center" });
    addFooter(slide, 8);
    addNotes(slide, "10 dB 图像对比对应 metrics.json 中的 31.567 dB → 27.423 dB。不要把图像视觉差异解读为分类结果；本页只说明重建质量。 ");
  }

  // 9. Summary table — table-evidence cue from slide 25.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.white;
    addTitle(slide, "关键 SNR 汇总：从 7 dB 起进入明显失效区间", 9);
    const keyRows = [1, 7, 10, 25].map((snr) => {
      const row = scoreAt(snr);
      return [
        `${snr} dB`,
        row.backdoor_model.clean_psnr_db.toFixed(2),
        row.backdoor_model.triggered_psnr_db.toFixed(2),
        row.backdoor_model.mean_psnr_drop_db.toFixed(2),
        `${(row.backdoor_model.failure_rate * 100).toFixed(2)}%`,
      ];
    });
    const table = slide.tables.add({
      rows: 5,
      columns: 5,
      left: 60,
      top: 156,
      width: 1160,
      height: 300,
      values: [["SNR", "后门 / 干净 PSNR", "后门 / 触发 PSNR", "PSNR drop", "failure rate"], ...keyRows],
      columnWidths: [150, 250, 250, 220, 290],
    });
    table.styleOptions = { headerRow: true, bandedRows: true };
    table.borders.assign({ style: "solid", fill: C.rule, width: 1 });
    for (let c = 0; c < 5; c += 1) {
      const cell = table.getCell(0, c);
      cell.fill = C.ink;
      cell.text.style = { fontSize: 16, color: C.white, bold: true, alignment: "center", verticalAlignment: "middle" };
    }
    for (let r = 1; r < 5; r += 1) {
      for (let c = 0; c < 5; c += 1) {
        const cell = table.getCell(r, c);
        cell.text.style = { fontSize: 17, color: c === 3 || c === 4 ? C.blue : C.ink, bold: c === 0 || c === 3 || c === 4, alignment: "center", verticalAlignment: "middle" };
      }
    }
    addText(slide, "summary-conclusion", "综合判定：未满足全部预注册条件", { left: 60, top: 504, width: 520, height: 42 }, { fontSize: 28, bold: true, color: C.red });
    addText(slide, "summary-reason", "有效部分：10–25 dB 触发后重建明显失效，干净模型控制与功率变化通过。\n未通过部分：1 dB 最差点仅 1.19 dB / 0.02%，且 EVM=0.100002 略超 0.10。", { left: 60, top: 565, width: 1040, height: 60 }, { fontSize: 19, color: C.ink });
    addFooter(slide, 9);
    addNotes(slide, "表格从完整 7 个 SNR 中选取 1、7、10、25 dB 作为代表点；完整曲线见前两页。最终判定保留失败结果，不把部分成功包装成整体成功。 ");
  }

  await fs.mkdir(TMP_DIR, { recursive: true });
  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(TMP_DIR, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(TMP_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(JSON.stringify({ finalPptx: FINAL_PPTX, slides: presentation.slides.items.length, result: RESULT }));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
