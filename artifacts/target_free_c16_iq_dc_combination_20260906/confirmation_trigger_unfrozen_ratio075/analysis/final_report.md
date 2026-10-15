# IQ+DC combination-specific confirmation

- Protocol: target-free c=16, SNR=10 dB, variant C, 80k steps, 3x3 training/poison seeds.
- Completed runs: 9/9. Feasible checkpoints: 0/9.
- Conclusion: **partial support only**; the full acceptance claim is not supported because no run satisfied all gates.

## 10 dB aggregate

- full_drop_db: mean 4.9155, SD 0.4527, range [4.0804, 5.4340]
- iq_only_drop_db: mean 0.1364, SD 0.0264, range [0.0907, 0.1781]
- dc_only_drop_db: mean 2.3503, SD 0.3348, range [1.8290, 2.8538]
- iq_sign_flip_drop_db: mean 0.9509, SD 0.1934, range [0.6872, 1.3171]
- dc_sign_flip_drop_db: mean 0.4330, SD 0.0492, range [0.3234, 0.4995]
- equal_evm_random_drop_db: mean 0.1916, SD 0.0011, range [0.1894, 0.1932]
- clean_model_any_drop_db: mean 0.5958, SD 0.0019, range [0.5934, 0.5985]
- utility_drop_db: mean -0.1120, SD 0.0151, range [-0.1334, -0.0856]
- removal_gap_iq_db: mean 2.5652, SD 0.1396, range [2.2514, 2.7138]
- removal_gap_dc_db: mean 4.7791, SD 0.4299, range [3.9897, 5.2739]
- interaction_contrast_db: mean 2.4288, SD 0.1199, range [2.1607, 2.5357]
- evm_rms: mean 0.0927, SD 0.0000, range [0.0926, 0.0927]
- power_change_db: mean -0.0018, SD 0.0002, range [-0.0021, -0.0015]

## Gate failures

The dominant failures are full attack drop below 6 dB and IQ-only/removal-gap-IQ constraints.
No test-set or multi-SNR result is reported because selection requires a feasible validation checkpoint first.
