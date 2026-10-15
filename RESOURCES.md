# DeepJSCC Resources

## Knowledge

- [Paper: “Deep Joint Source-Channel Coding for Wireless Image Transmission” — Bourtsoulatze, Kurka & Gündüz](https://arxiv.org/abs/1809.01733)
  DeepJSCC 基础论文。用于系统模型、功率归一化、AWGN/慢瑞利信道、MSE 训练和 graceful degradation。
- [Paper: “DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback” — Kurka & Gündüz](https://arxiv.org/abs/1911.11174)
  用于理解反馈如何把一次传输扩展为多阶段 refinement，以及固定长度与可变长度传输。
- [Paper: “Bandwidth-Agile Image Transmission with Deep Joint Source-Channel Coding” — Kurka & Gündüz](https://arxiv.org/abs/2009.12480)
  用于理解渐进式传输、successive refinement 和 multiple descriptions。
- [Open-access journal record — UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10089480/)
  原始 2019 IEEE TCCN 论文的开放获取记录、正式书目信息与作者接受稿入口。
- [Paper: “A Channel-Triggered Backdoor Attack on Wireless Semantic Image Reconstruction” — Wan et al.](https://arxiv.org/abs/2503.23866)
  与本课最接近的信道触发 DeepJSCC 后门论文。用于区分信道触发、模型特异性和目标重建；本仓库的 IQ/DC 无目标可用性攻击不是它的直接复现。

## Wisdom (Communities)

- [IEEE Information Theory Society](https://www.itsoc.org/)
  用于跟踪 JSCC、有限码长和语义通信领域的学术活动与专业讨论。

## Gaps

- 后续课程需要补充一篇高质量综述，用于建立 DeepJSCC 各分支的统一分类。
- 尚无本地记录的论文直接验证“DeepJSCC 图像重构 + IQ imbalance + DC offset + 无目标可用性投毒”；仓库方法应作为待实验验证的研究假设，而不是已有文献结论。
