# 基于 liquid-dsp 的原始频域符号 OFDM 系统实施计划

## 1. 目标与边界

在 Ubuntu 主机 `seu@192.168.31.190` 上构建离线 OFDM 收发系统：

- C99/CMake 数据面，链接系统已安装的 liquid-dsp。
- FastAPI + React 控制面，支持上传二进制、选择内置 clean/triggered 样例、配置参数、运行任务、查看指标和下载产物。
- 数据子载波直接承载输入 `complex64`，不进行 BPSK/QAM 数据调制、信道编码、CRC 或字节打包。
- 使用 Wi-Fi 4 的 64 点子载波布局、16 点循环前缀、传统短/长训练序列和 4 个频域 pilot。
- 完成帧检测、定时/CFO 校正、LS CSI 估计、pilot 相位跟踪和 ZF 均衡；MMSE 作为可选模式。
- 系统保持纯离线文件处理，不包含 SDR、OTA 发射或真实无线设备控制。

## 2. OFDM 数据面

在 `ofdm/` 中实现独立库和 `ofdm_cli`。

### 2.1 输入与映射

- 输入格式固定为 little-endian、I/Q 交织的 `float32`，每 8 字节表示一个复数频域值。
- 拒绝空文件、非 8 字节整数倍文件以及包含 NaN/Inf 的输入。
- 每 64 个输入值作为一个逻辑块保存元数据，但数据按连续流映射到每个 OFDM 符号的 48 个数据子载波。
- 对 1024 点输入生成 22 个数据 OFDM 符号；最后一个符号有 32 个有效数据值和 16 个补零值，接收时按 manifest 截回 1024 点。
- FFT 为 64 点，CP 为 16 点。
- 占用子载波为 `k=-26...-1,+1...+26`。
- pilot 子载波为 `k={-21,-7,+7,+21}`。
- 数据子载波为占用子载波中除去四个 pilot 的 48 个位置。
- DC 和其余保护子载波固定为零。

### 2.2 帧结构与同步

- 使用 IEEE 802.11a/n legacy STF、LTF 和 pilot polarity 序列。
- 已知训练和 pilot 可以使用 `+1/-1`；输入数据本身不做任何星座映射。
- 帧结构为 `L-STF + 双 L-LTF + 原始数据 OFDM 符号`，不加入 SIGNAL、MAC 头或编码字段。
- STF 用于帧相关检测、粗定时和粗 CFO 估计。
- 双 LTF 用于精定时、残余 CFO 和逐占用子载波 LS 信道估计。
- 每个数据符号使用四个 pilot 估计并补偿公共相位误差。
- 默认执行带除零下限的 ZF 均衡；可选按噪声估计执行 MMSE。
- 按 manifest 中的有效复数点数去除 padding，并恢复输入顺序。

### 2.3 liquid-dsp 使用方式

- 使用 liquid-dsp 的 FFT、NCO、滤波和信道组件。
- 不使用会重新调制、增加帧头或 FEC 的 `ofdmflexframe`。
- 编译前检测系统 liquid-dsp 的头文件、库版本和 `pkg-config` 信息，并将版本写入 manifest。

### 2.4 CLI 模式

- `tx`：从频域输入生成时域波形及 manifest。
- `rx`：读取外部时域波形及 manifest，恢复频域数据和 CSI。
- `loopback`：依次执行 TX、可配置仿真信道和 RX。

CLI 使用 JSON 配置，字段至少包括：

- `mode`
- `input_path`
- `rx_waveform_path`
- `expected_complex_symbols`
- `snr_db`
- `cfo_rad_per_sample`
- `channel_taps`
- `seed`
- `equalizer`，取值为 `zf` 或 `mmse`
- `output_dir`

仿真信道支持 AWGN、有限长多径、CFO 和固定随机种子。默认受损测试使用 25 dB SNR、`0.001 rad/sample` CFO，以及最大时延小于 CP 的固定三径信道。

## 3. 输出接口

每次运行生成：

- `tx_waveform.bin`：发射端时域交织 float32 IQ。
- `rx_waveform.bin`：经过信道或外部输入的时域交织 float32 IQ。
- `recovered_symbols.bin`：恢复后的原始频域复数值。
- `csi.bin`：形状为 `[数据 OFDM 符号数, 52]` 的有效子载波复数 CSI。
- 与恢复符号和 CSI 对应的 CSV 检查文件。
- `manifest.json`：格式、有效长度、补零数、子载波映射、参数、liquid-dsp 版本和文件 SHA-256。
- `metrics.json`：检测状态、CFO 真值及估计、CSI NMSE、均衡前后 EVM、样本数和错误信息。

所有复数 BIN 文件均使用 little-endian、I/Q 交织 float32。JSON 中必须记录每个 BIN 的维度和轴语义。

## 4. Web 控制面

在 `web/backend/` 和 `web/frontend/` 中实现。

### 4.1 FastAPI 后端

提供以下接口：

- `GET /api/samples`
- `POST /api/uploads`
- `POST /api/jobs`
- `GET /api/jobs/{id}`
- `GET /api/jobs/{id}/artifacts/{name}`

行为要求：

- 只允许浏览器上传文件或选择 clean/triggered 两份内置样例，不开放任意服务器路径。
- 校验上传扩展名、大小、二进制结构和有限数值，并生成后端控制的安全文件名。
- 使用单后台任务队列调用 `ofdm_cli`，捕获标准输出、标准错误和退出码。
- 任务状态原子写入磁盘；刷新页面后可恢复状态。
- 服务重启时，将遗留的运行中任务标记为 `interrupted`，不自动重复执行。
- 任务输出限制在后端创建的独立目录中，禁止路径穿越和任意命令参数。

### 4.2 React 前端

页面提供：

- clean/triggered 内置样例选择或本地 BIN 上传。
- `loopback`、`tx`、`rx` 模式选择。
- SNR、CFO、多径 taps、随机种子、ZF/MMSE 参数控制。
- 当前任务状态、日志、帧检测、CFO、CSI 和 EVM 指标。
- CSI 幅度/相位、均衡前后散点图和 EVM 对比。
- BIN、CSV、JSON 产物下载。

React 构建后的静态资源由 FastAPI 提供。默认部署到可信局域网的 `0.0.0.0:8000`，限制 CORS、上传大小和输出目录，不暴露 shell 或服务器文件浏览能力。

## 5. 验证与验收

### 5.1 数据面测试

- 验证每个符号严格包含 48 data、4 pilot 和 12 null。
- 验证 1024 点输入产生 22 个数据符号，并准确去除最后 16 个 padding。
- 使用随机复数、冲激输入、clean 样例和 triggered 样例完成 TX→RX。
- 理想链路要求恢复点数完全一致、最大绝对误差 `<1e-4`、EVM `<-70 dB`。
- 默认受损链路必须检测到帧并恢复全部点。
- 受损链路 CFO 估计误差 `<1e-4 rad/sample`。
- 均衡后 EVM 至少改善 10 dB，且低于 -15 dB。
- CSI NMSE 低于 -15 dB。

### 5.2 异常与 Web 测试

- 覆盖空文件、截断 IQ、NaN/Inf、RX 缺少 manifest、超限上传、非法 channel tap 和任务冲突。
- 使用 pytest 测试 FastAPI 接口。
- 使用 React Testing Library/Vitest 测试参数表单、状态和错误展示。
- 增加一次从上传、提交任务到下载产物的端到端测试。

## 6. 部署步骤

1. 配置 `seu@192.168.31.190` 的 SSH 公钥认证；不得在聊天或仓库中传递私钥内容。
2. 复检主机名、用户、远端项目目录、编译器、CMake、Python、Node 和 liquid-dsp 版本。
3. 远端项目根目录默认使用 `/home/seu/deepjscc_backdoor`；若已有仓库则保留其未提交修改，不执行破坏性同步。
4. 本地实现和静态检查 C、Python、React 代码，再将新增模块和两份小型样例同步到服务器。
5. 在服务器用 CMake 构建 C 数据面，构建 React 静态资源，并启动 FastAPI 集成测试。
6. 用两份内置 BIN 依次完成理想和默认受损链路验收，保存配置、指标、日志与哈希。

## 7. 固定假设

- 首版按 Wi-Fi 4 的 20 MHz、64 点 FFT 语义处理。
- CFO 内部与 manifest 均使用 `rad/sample`。
- Web 服务部署在可信局域网内；首版不实现用户账户系统。
- 当前任务属于短时 CPU 离线仿真，不使用 GPU。
- 当前 SSH 预检返回 `Permission denied (publickey,password)`；认证恢复前不得反复尝试或猜测其他凭据。
