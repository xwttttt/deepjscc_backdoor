# Encoder 联合训练使 latent 非平稳

用户已识别出：当 encoder 与 decoder 联合更新时，即使训练图像和 poison mask 固定，同一图像的 latent `z=Eθ(x)` 也会随 step 改变。这意味着后门训练是在动态表示空间中让模型与 trigger 共同适应，后续应进一步区分固定输入数据、动态 latent 和动态 trigger 参数。
