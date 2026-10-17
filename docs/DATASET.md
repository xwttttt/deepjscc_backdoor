# DIV2K 数据与训练

## 1. 保留的数据处理逻辑

用户提供的三个脚本在：

- `src/src/make_train_lmdb_dataset.py`：训练图随机裁 4 个 256×256 图块，valid 每图裁 1 个；
  Cosmos CV16 编解码后，保存 `image` 和 `token` 到连续九位 ASCII key 的 LMDB。
- `src/src/make_train_lmdb_dataset.sh`：原环境启动示例，保留原文件；其中 Conda 和
  `scripts/make_train_lmdb_dataset.py` 路径属于原 CTJSCC 工程，不能直接在本仓库原样运行。
- `src/src/data_provider.py`：原 image/token DataProvider，保留其接口和归一化。

本 DeepJSCC 训练使用 `scripts/div2k_lmdb_dataset.py` 适配：只消费已生成的 image，
裁剪到 `[0,1]`，不消费 token，不重新读 PNG 或动态裁剪。不能直接将原 Provider 的
`[-1,1]` 输出接给当前 sigmoid RGB DeepJSCC；因此适配读取层不能省略。

## 2. 数据分区

| 数据 | 数量 | 用途 |
| --- | --- | --- |
| 官方 train HR | 800 张 | 4 crops/image → 3200 条固定 train LMDB |
| 官方 valid HR | 100 张 | 1 crop/image → 100 条固定 valid LMDB |
| valid LMDB 键 0–49 | 50 条 | 训练验证和选模 |
| valid LMDB 键 50–99 | 50 条 | 独立测试/收发；不用于选模 |

Git 已包含测试导出 `data/div2k_cv16_test/{test.npy,manifest.json}`。
**源 HR 压缩包、完整 train/valid LMDB 和 Cosmos 权重不在 Git 中**；数据来源、
训练服务器已有路径和历史 data.mdb 哈希在 `data/dataset_sources.json`。
不能把 50 条测试导出当成完整训练集。大数据与权重保留服务器，不在本机下载或重建。

## 3. 优先使用训练服务器已有固定 LMDB

GPU 服务器当前已有数据路径记录为：

```text
/home/gxhen/hylan/CTJSCC/datasets/DIV2K/train_256_CV16_float32.lmdb
/home/gxhen/hylan/CTJSCC/datasets/DIV2K/valid_256_CV16_float32.lmdb
```

在该服务器仓库根目录建立**不存在的**目标链接（不要覆盖已存在目录）：

```bash
mkdir -p data/DIV2K/lmdb
ln -s /home/gxhen/hylan/CTJSCC/datasets/DIV2K/train_256_CV16_float32.lmdb data/DIV2K/lmdb/
ln -s /home/gxhen/hylan/CTJSCC/datasets/DIV2K/valid_256_CV16_float32.lmdb data/DIV2K/lmdb/
```

也可修改新训练配置的 `[dataset]` 为服务器实际路径，不必复制完整 LMDB。

## 4. 需要重新制作时：仍调用原脚本

只在专用 GPU 服务器运行，先准备 CTJSCC 工程的 `lib.cosmos_tokenizer` 和其
`COSMOS_TOKENIZER_CKPT_DIR` 对应的 CV8x16x16 encoder/decoder JIT 权重。
它们是外部依赖，本仓库不伪造或动态下载 tokenizer 实现。

```bash
# 下载官方 train/valid HR，保留源压缩包；不在本机执行大下载：
python3 scripts/download_div2k.py --root data/DIV2K --split all --keep-archives

# 依赖安装在制作数据的 GPU 环境，torch/torchvision 需匹配：
python3 -m pip install lmdb loguru tqdm torchvision

# 先阅读并检查 CTJSCC 工程及 tokenizer 权重，再运行原 maker：
PYTHON=/path/to/gpu/python \
  bash scripts/prepare_lmdb.sh /path/to/CTJSCC /path/to/new_CV16_lmdb --seed 42
```

包装脚本设置导入路径并固定 CV16/256/4-crops/float32，拒绝复用既有输出目录，
因为原 maker 的 `create_dataset()` 会清除旧记录。`DATASET_ROOT` 默认是本仓库 `data`，
它下面应有 `DIV2K/DIV2K_train_HR`、`DIV2K/DIV2K_valid_HR`。
新生成的随机图块、tokenizer 版本或权重未必与历史固定 LMDB 相同；必须重新记录
来源/哈希并重新训练，不能沿用历史 checkpoint 的效果报告。

## 5. 正常模型训练和后门模型训练

使用 Python 3.11+、CUDA GPU 服务器；默认路径见上节。

```bash
# 正常模型：从交付 P0 继续训练，不宣称从零复现原 P0：
python3 scripts/run_div2k_clean_p0.py --init-checkpoint models/p0_clean.pt \
  --output-dir runs/p0_finetune_new --steps 5000 --batch-size 4 --mixed-precision

# 指定 stronger_struct 原配置：
PYTHON=/path/to/gpu/python bash scripts/train_stronger.sh runs/stronger_new
```

如需用新 P0 训练后门，直接调用入口并明确 checkpoint：

```bash
python3 scripts/run_div2k_two_drop.py \
  --config configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml \
  --clean-checkpoint runs/p0_finetune_new/best.pt --output-root runs/stronger_from_new_p0
```

本次不修改训练目标：保留交付 stronger 模型的 `localized_structure` 版本，
不能称为旧 whole-frame PSNR MaxDrop 的逐字复现。只有 clean drop 与 triggered
局部结构 drop 参与该版优化/选模，不增添 EVM/功率/感知项。
历史验证 clean drop 超过门槛的事实仍保留，不能把该模型写成已通过 clean gate。

## 6. 固定测试导出与 OTA

```bash
python3 scripts/export_div2k_lmdb_test.py \
  --valid-lmdb /path/to/valid_256_CV16_float32.lmdb --output-dir /path/to/new_test_export
```

只导出 valid 键 50–99：clip 到 `[0,1]`、乘 255 四舍五入成 uint8 NCHW，
生成 NPY 和来源/输出哈希 manifest。两文件一起复制到 Tx 主机。
部署模型必须是相同训练数据来源的模型；Encoder 和 Decoder 用同一个 checkpoint。
随后按 [WORKFLOW.md](WORKFLOW.md) 分步骤收发；Rx 主机无需 LMDB/源图片/Tokenizer。
