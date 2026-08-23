# 文献证据矩阵

更新日期：2026-08-23。`证据`表示原文直接支持的结论；`本项目用途`是迁移设计，不等同于
论文作者已在 IQ imbalance + DC offset 的 DeepJSCC 组合上验证。

## 检索与筛选协议

检索源包括 arXiv、IEEE DOI/作者机构页面和 GitHub；组合关键词为 `DeepJSCC/semantic
communication` × `backdoor/Trojan/adversarial/channel-triggered`，以及 `RF fingerprint`
× `IQ imbalance/DCFO/CFO/backdoor`。优先纳入原始论文、作者稿、正式 DOI 和官方仓库；
二手页面只用于定位，关键结论回到原文摘要或作者页面核验。纳入与图像重构、语义通信、
RF 指纹硬件失真或无线对抗约束直接相关的工作；排除只有泛化 CV 后门、无可核验出处、
以及与复数基带信号无关的工作。正式系统综述应另行保存数据库查询式、命中数和筛选流程。

| 方向 | 文献 | 证据 | 本项目用途/限制 |
|---|---|---|---|
| DeepJSCC | Bourtsoulatze et al., *Deep Joint Source-Channel Coding for Wireless Image Transmission*, 2019, [arXiv](https://arxiv.org/abs/1809.01733), [DOI](https://doi.org/10.1109/TCCN.2019.2919300) | 端到端图像 JSCC 在 AWGN/衰落下联合优化 | 干净基线；不是安全论文 |
| 信道后门 | Wan et al., *A Channel-Triggered Backdoor Attack on Wireless Semantic Image Reconstruction*, 2025/2026 v3, [arXiv](https://arxiv.org/abs/2503.23866) | 以衰落分布或噪声 PSD 作为触发，研究目标图像重构，并在多种 JSCC 架构上评估 | 最直接的方法学母线；未发现官方代码，论文结果不能当作本项目复现结果 |
| 重构后门 | Zhou et al., *BASS*, ICC 2024, [arXiv](https://arxiv.org/abs/2404.13279), [DOI](https://doi.org/10.1109/ICC51166.2024.10622193) | 针对高维语义符号重构提出攻击与逆向/剪枝防御 | 定义目标重构及防御对照；分类式防御需重新验证 |
| 语义后门 | Sagduyu et al., CISS 2023, [arXiv](https://arxiv.org/abs/2212.11205), [DOI](https://doi.org/10.1109/CISS56502.2023.10089692) | 训练投毒可在语义通信的图像/潜变量/接收信号阶段植入 Trojan | 威胁模型和 SNR/信道使用次数消融 |
| 多域对抗 | Sagduyu et al., IEEE Communications Magazine 2023, [arXiv](https://arxiv.org/abs/2212.10438), [DOI](https://doi.org/10.1109/MCOM.006.2200878) | 区分图像域和无线域测试时对抗扰动 | 帮助区分“后门”与“仅测试时攻击” |
| 物理层对抗 | Nan et al., IEEE JSAC 2023, [arXiv](https://arxiv.org/abs/2305.07220), [DOI](https://doi.org/10.1109/JSAC.2023.3288249) | SemAdv 生成物理层对抗扰动，SemMixed 用混合对抗训练增强鲁棒性 | 对抗基线与 impairment augmentation 防御 |
| 反证/鲁棒性 | Zhang et al., *Unanticipated Adversarial Robustness of Semantic Communication*, 2026, [arXiv](https://arxiv.org/abs/2603.24082) | 噪声训练可诱导解码器平滑；提出渐进梯度上升攻击并报告 DeepJSCC 的较高攻击功率需求 | 必须报告失败和最小攻击功率，避免预设“必然易攻” |
| 多用户无线后门 | Sagduyu et al., *Wireless Backdoor Attack and Defense for Semantic Communications over Multiple Access Channel*, 2026, [arXiv](https://arxiv.org/abs/2606.30595) | 在共享接收信号上研究低功率 OTA 触发和触发感知训练 | 后续扩展；不纳入第一阶段单链路实验 |
| RF 指纹注入 | Gu et al., *PR-RFFI*, IEEE TMC 2026, [DOI](https://doi.org/10.1109/TMC.2026.3665125), [作者列表](https://xiaolingu.github.io/publication/) | 将 IQ imbalance 注入 Wi-Fi STF，并引入温度不敏感 DCFO；原文报告保持通信质量并提高设备区分度 | 触发物理含义来源；协议结构与 DeepJSCC 不同，是最大外推风险 |
| RF 指纹基础 | Zhang et al., *RFFI for Narrowband Systems, Modelling and Classification*, IEEE TIFS 2021, [DOI](https://doi.org/10.1109/TIFS.2021.3088008), [机构页面](https://livrepository.liverpool.ac.uk/id/eprint/3124282) | 系统建模 IQ imbalance、PA、频偏等；指出频偏受温度影响且接收端失真需校准 | 参数建模、温漂和接收机混淆控制 |
| RF 后门 | Zhao et al., *Stealthy Backdoor Attack on RF Signal Classification*, ICCCN 2023, [DOI](https://doi.org/10.1109/ICCCN58024.2023.10230152), [作者 PDF](https://www.winlab.rutgers.edu/~yychen/daisylab/papers/Stealthy%20Backdoor%20Attack%20on%20RF%20Signal%20Classification.pdf) | 联合优化 RF 触发与分类模型并加入隐蔽性损失 | EVM/MSE 正则和触发消融；结果仅限分类，不可外推到重构 |
| RF 指纹后门 | Zhao et al., *Explanation-Guided Backdoor Attacks Against Model-Agnostic RF Fingerprinting Systems*, IEEE TMC 2025（online 2024）, [DOI](https://doi.org/10.1109/TMC.2024.3487967), [机构页面](https://livrepository.liverpool.ac.uk/id/eprint/3186204) | 面向 RF 指纹系统的模型无关、解释引导触发设计 | RF 指纹领域的攻击参照；不等于信道触发 DeepJSCC |
| RF 对抗工具 | Flowers et al., *Evaluating Adversarial Evasion Attacks in Wireless Communications*, IEEE TIFS 2020, [DOI](https://doi.org/10.1109/TIFS.2019.2934069), [代码](https://github.com/brysef/rfml) | 通信约束的 FGSM/PGD、EVM/频谱约束与对抗训练 | 对抗基线与物理隐蔽指标 |

## 证据缺口

- 截至检查日期，未发现 CT-BA、PR-RFFI 或 CQP-RFFI 的官方公开实现。
- 尚无直接论文同时验证“DeepJSCC 图像重构 + IQ imbalance + DCFO + 训练投毒”。这正是
  研究空缺，也是不能把相邻工作结果拼接成既成事实的原因。
- PR-RFFI 全文的 DCFO 提取与注入细节需通过学校数据库/作者版本核验；本仓库只实现明确
  标注的连续相位代理。
- 2026 年论文仍较新，应在正式投稿前重新检索版本、勘误、代码和后续独立复现。
