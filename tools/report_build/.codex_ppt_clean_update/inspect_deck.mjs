import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const input = 'C:/Users/Win 10 Pro/Documents/ChatGPT/deepjscc_backdoor/deepjscc_c16_target_free_reconstruction_failure_visual_report.pptx';
const outDir = 'C:/Users/Win 10 Pro/Documents/ChatGPT/deepjscc_backdoor/.codex_ppt_clean_update';
async function writeBlob(path, blob) {
  await fs.writeFile(path, Buffer.from(await blob.arrayBuffer()));
}
const deck = await PresentationFile.importPptx(await FileBlob.load(input));
const snapshot = await deck.inspect({
  kind: 'deck,slide,textbox,shape,image,table,chart,notes,layout',
  include: 'id,slide,name,title,text,textPreview,bbox,bboxUnit,chartType,alt,isPlaceholder,placeholders',
  maxChars: 30000,
});
await fs.writeFile(`${outDir}/before-inspect.ndjson`, snapshot.ndjson, 'utf8');
for (const [i, slide] of deck.slides.items.entries()) {
  const png = await deck.export({ slide, format: 'png', scale: 1 });
  await writeBlob(`${outDir}/before-slide-${i + 1}.png`, png);
  const layout = await slide.export({ format: 'layout' });
  await fs.writeFile(`${outDir}/before-slide-${i + 1}.layout.json`, await layout.text(), 'utf8');
}
const montage = await deck.export({ format: 'webp', montage: true, scale: 1 });
await writeBlob(`${outDir}/before-montage.webp`, montage);
console.log(snapshot.ndjson);
