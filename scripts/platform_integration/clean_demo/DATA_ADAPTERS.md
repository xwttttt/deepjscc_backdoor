# 分机 Tx/Rx：固定 Cosmos-LMDB 数据集

本入口沿用用户提供的 `make_train_lmdb_dataset.py`、`.sh`、`data_provider.py` 所定义的预处理来源：DIV2K HR 裁剪、Cosmos CV16 encode/decode、LMDB 图像与 token。现有 LMDB **不重新生成**。发送使用 valid LMDB 键 50–99 离线导出的固定 `test.npy`（`uint8 [50,3,256,256]`）和相邻 `manifest.json`；编码时一次读取并验证哈希，不动态加载、裁剪原图或重新运行 tokenizer。DeepJSCC 按训练约定将 uint8 转为 `[0,1]`，不使用 `data_provider.py` 面向其他模型的 `[-1,1]` 归一化。

无线接口只走平台原有 `TX.py` / `RX.py`。Encoder/Decoder 仍需要相应的 PyTorch 模型定义、checkpoint、NumPy；这是模型推理依赖，不是额外的收发器。`clean_demo/run_tx_data.py`、`run_rx_decode.py` 是旧适配入口，**不用于本分机流程**。

## Tx 主机

先把 `test.npy`、`manifest.json`、匹配的 checkpoint、`src/src/encode_prepared_npy.py`、`src/src/deepjscc/model.py`、`src/src/channels.py` 与支持 `--input-bin` 的 `TX.py` 放到主机。模型环境需 `torch`、`numpy`、`einops`；`TX.py` 文件模式需 `pyzmq`。建议先只生成文件，不发射：

```bash
python3 src/src/encode_prepared_npy.py \
  --dataset data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy \
  --checkpoint models/div2k_lmdb_cv16_20260923/p0_clean.pt \
  --output-dir captures/tx_prepared_NEW_RUN --max-images 1
```

输出 `encoder/burst_00000001.cf32le` 是功率归一化的频域 complex64 latent，每个复数按小端 `float32 I,Q` 交织，不是完整时域 OFDM 波形。`manifest.json` 记录预处理导出文件、checkpoint、latent 的哈希及逐图块 burst ID。数据身份与训练 checkpoint 必须一致，不能换成旧的原图裁剪 NPY。

在现场确认设备连接、频率、天线、增益、授权以及 TX 服务已启动后，才能显式加 `--send`；这会逐文件调用 **`TX.py --input-bin`**，不会由编码脚本直接发 ZeroMQ：

```bash
python3 src/src/encode_prepared_npy.py \
  --dataset data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy \
  --checkpoint models/div2k_lmdb_cv16_20260923/p0_clean.pt \
  --output-dir captures/tx_send_NEW_RUN --max-images 1 \
  --tx-script ./TX.py --endpoint tcp://127.0.0.1:5555 --send
```

Git 仓库 `ofdm-based` 分支的原始 `TX.py` **只有 fake 样本模式**，不支持 `--input-bin`；必须先部署经过文件输入适配的 `TX.py`。编码入口在发送前检查其 `--help`，不支持便拒绝发送。`tx_invoked=true` 仅表示 TX 客户端调用成功，不证明实机空口送达。

## Rx 主机

原 `RX.py` 作为唯一 RX ZeroMQ 消费者，不作修改。它把每 burst 的 JSON、64-bin CSI、均衡后的 `.data.cf32le` 以及批次 `.npy` 存盘。接收完成后，以其固定 NPY 目录调用现有 decoder-only 脚本：

```bash
python3 RX.py --endpoint tcp://127.0.0.1:5556 \
  --save-dir captures/rx_NEW_RUN --npy-dir captures/rx_NEW_RUN/npy \
  --max-bursts 1
python3 src/src/decode_rx_npy.py captures/rx_NEW_RUN/npy \
  --checkpoint models/div2k_lmdb_cv16_20260923/p0_clean.pt \
  --output-dir captures/reconstructed_NEW_RUN --max-records 1 --device cpu
```

Decoder 需要 `torch`、`numpy`、`Pillow`，不依赖 `deepjscc_codec.py`、`clean_demo` 或另一个 ZeroMQ 接收器。保留 RX JSON/CSI 与解码 PNG；批次 NPY 只包含数据，不带逐条 burst ID，需按接收顺序与 RX JSON 对照。Anchor CRC 校验控制信息，不保证原始 IQ payload 逐字节正确。

## 状态

本地已完成编码文件入口、`TX.py` 文件模式和文档；静态编译通过。当前 Windows Python 未安装 `torch`，未完成本地模型推理。xwt SSH 尚不可达，**未同步到服务器，也未做真实发射/接收验证**。之前的仿真指标不能视为本入口的验证结果。
