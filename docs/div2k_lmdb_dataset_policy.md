# 后续实验的数据集策略：DIV2K Cosmos LMDB

从本决策起，后续**新实验**统一使用预生成、固定不变的 DIV2K image/token LMDB。输入逻辑以用户提供的 `make_train_lmdb_dataset.py`、`make_train_lmdb_dataset.sh`、`data_provider.py` 为参考：在离线制作阶段对 DIV2K HR 原图裁剪，执行 Cosmos tokenizer `encode`、`decode`，将**解码后的图像**及对应 token 写入 LMDB；训练阶段从 LMDB 读取，不动态加载或裁剪原图。

这与旧的 `data/DIV2K/prepared_256_seed_20260923/*.npy` 不是同一数据集：旧数据存原始图像裁剪，新 LMDB 存 tokenizer 解码后的图像。旧 checkpoint、50 图 OFDM 仿真和重建图仍按旧数据身份保留，不重命名为 LMDB 实验结果。新训练、验证与收发对照都必须记录 LMDB 版本、构建参数、源图像及 tokenizer checkpoint 哈希、样本数量和数据划分；不得混用旧 `.npy` 结果作新模型的指标。

专用服务器已有 CTJSCC 制作的 CV16 LMDB：`/home/gxhen/hylan/CTJSCC/datasets/DIV2K/train_256_CV16_float32.lmdb` 与 `valid_256_CV16_float32.lmdb`，分别含 3200 和 100 个样本。项目的 `data/DIV2K/lmdb/` 链接指向这两份现有库；**不重新生成或清空它们**。验证使用 valid 键 0–49，留出测试使用键 50–99。读取入口为 `scripts/div2k_lmdb_dataset.py`，训练配置为 `configs/div2k_two_drop_struct_lmdb_256.toml`。

服务器 `data.mdb` SHA-256：train `c6fc0c7307d107f8d3867177e0569757f3a9ba1193a5210a097371f6755a1c75`；valid `ed33a16a4c9b59e5105f834a2a760d907772c58334b37feb274a5cc79634b738`。附件中的生成 Python 脚本和读取器与服务器 CTJSCC 对应文件哈希一致；shell 启动脚本哈希不同，因此生成环境仍以已存在 LMDB 的内容哈希为准。

DeepJSCC 使用 `[0,1]` RGB 输入和 sigmoid RGB 输出；Cosmos 解码图像存在少量超出该范围的值。因此 DeepJSCC 适配层把读取出的图像裁到 `[0,1]`，不使用原附带 `data_provider.py` 的 `[-1,1]` 归一化。token 仍在 LMDB 中保存，但现有 DeepJSCC 图像输入训练不消费 token。这个适配与原 CTJSCC 的训练语义不同，须作为新实验单独报告。

OFDM 主机目前没有 `lmdb` Python 包。为保持其现有发射脚本不依赖新包，使用 `scripts/export_div2k_lmdb_test.py` 从 valid LMDB 键 50–99 离线导出固定的 uint8 `test.npy`，同时保存源 LMDB 与导出文件哈希。这个 `.npy` 是 **LMDB 测试集的派生视图**，不是旧原图裁剪 `.npy`；禁止把二者混用或复用旧实验结果。

已在专用服务器生成该派生测试集：`data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy`，SHA-256 为 `83f35a532a6a1240499262e208c7cacb055a94b1c67f65fe5a107a4e6ea05445`。P0 的 1-step 和后门的 10-batch GPU 冒烟测试均通过（后者覆盖 4 个投毒样本）；对应指标快照在 `results/div2k_lmdb_cv16_20260923_smoke/`，不是正式模型指标。

2026-09-23 的完整 LMDB 运行已完成：P0 5000 steps，留出测试 PSNR 38.300 dB；v2 后门 40 epochs，选中 epoch 29，留出测试 clean drop 0.122 dB、局部结构 drop 14.814 dB、整帧 triggered drop 16.945 dB（100 dB 近无损信道代理）。50 图 OFDM **数字仿真**为 100/100 CRC、clean PSNR 41.835 dB、triggered PSNR 25.055 dB、两者相差 16.781 dB。指标与图片见 `results/div2k_lmdb_cv16_20260923_full/`；checkpoint 和大日志只保留在服务器。该结果不是 OTA。示例触发重建呈现较大范围的细密纹理，不能仅凭高局部结构分数声称形成了肉眼可见的局部崩塌。此外，原 LMDB 制作流程未记录逐样本裁剪坐标与源图像清单，因此现有库可由哈希锁定，但不能保证从源 PNG 精确重建同一 LMDB。

2026-09-28 在专用 GPU 服务器完成一轮更强视觉失真候选训练：配置
`configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml`，服务器输出目录
`runs/div2k_lmdb_cv16_20260928/stronger_struct_60ep_001/`。仍使用同一固定 LMDB、
P0、5% 投毒和纯双 drop 损失；提高结构幅度封顶与触发目标并延长训练。
留出测试 50 图的 clean drop 为 0.463 dB、局部结构 drop 为 19.729 dB、
整帧 triggered drop 为 28.165 dB；同一 LMDB 旧模型分别为 0.122、14.814、
16.945 dB。验证集 clean drop 为 0.710 dB，超过配置门槛 0.5 dB，
因此仍属候选。六图直接推理对照见
[`results/div2k_lmdb_cv16_20260928_stronger/`](../results/div2k_lmdb_cv16_20260928_stronger/README.md)；
没有经过 OFDM 或实机 RF。
