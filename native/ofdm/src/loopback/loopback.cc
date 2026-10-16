#include "loopback/loopback.h"
#include <cmath>
#include <random>
#include <algorithm>

namespace ofdm {

LoopbackResult run_loopback(const cf32* data, size_t len, uint32_t burst_id,
                            const LoopbackConfig& cfg) {
    // TX
    TxPipeline tx;
    auto tx_result = tx.build_burst(data, len, burst_id);

    // Channel
    auto& tx_samples = tx_result.samples;
    std::vector<cf32> rx_samples;
    rx_samples.reserve(tx_samples.size() + 256);

    // Random delay
    std::mt19937 rng(42);
    int delay = 0;
    if (cfg.random_delay > 0) {
        std::uniform_int_distribution<int> dist(0, cfg.random_delay - 1);
        delay = dist(rng);
        rx_samples.resize(delay, cf32(0, 0));
    }

    // Simple multipath: h = [1, 0.3*exp(j*pi/4), 0.1*exp(j*pi/2)]
    if (cfg.multipath) {
        std::vector<cf32> channel = {
            cf32(1.0f, 0.0f),
            cf32(0.3f * std::cos(M_PI / 4.0f), 0.3f * std::sin(M_PI / 4.0f)),
            cf32(0.1f * std::cos(M_PI / 2.0f), 0.1f * std::sin(M_PI / 2.0f))
        };
        rx_samples.resize(delay + tx_samples.size() + channel.size() - 1, cf32(0, 0));
        for (size_t n = 0; n < tx_samples.size(); n++) {
            for (size_t k = 0; k < channel.size(); k++) {
                rx_samples[delay + n + k] += tx_samples[n] * channel[k];
            }
        }
    } else {
        rx_samples.insert(rx_samples.end(), tx_samples.begin(), tx_samples.end());
    }

    // CFO
    if (std::abs(cfg.cfo_hz) > 0.01f) {
        float cfo_rad = cfg.cfo_hz * 2.0f * M_PI / (float)SAMPLE_RATE;
        for (size_t i = 0; i < rx_samples.size(); i++) {
            float phase = (float)i * cfo_rad;
            rx_samples[i] *= cf32(std::cos(phase), std::sin(phase));
        }
    }

    // AWGN
    if (cfg.awgn_snr_db < 100.0f) {
        // Compute signal power
        float sig_power = 0;
        for (auto& s : rx_samples) sig_power += std::norm(s);
        sig_power /= rx_samples.size();
        float noise_power = sig_power * std::pow(10.0f, -cfg.awgn_snr_db / 10.0f);
        float noise_sigma = std::sqrt(noise_power / 2.0f);

        std::normal_distribution<float> dist(0.0f, noise_sigma);
        for (auto& s : rx_samples) {
            s += cf32(dist(rng), dist(rng));
        }
    }

    // RX
    RxPipeline rx;
    auto results = rx.process(rx_samples.data(), (int)rx_samples.size());

    LoopbackResult lb;
    lb.success = false;

    if (results.empty()) return lb;

    // Find matching burst
    auto& best = results[0];
    lb.success = true;
    lb.rx_meta = best.meta;
    lb.recovered = best.data;

    // Compute error metrics
    size_t compare_len = std::min(len, best.data.size());
    float max_err = 0;
    double err_sum = 0, sig_sum = 0;
    for (size_t i = 0; i < compare_len; i++) {
        float e = std::abs(best.data[i] - data[i]);
        max_err = std::max(max_err, e);
        err_sum += std::norm(best.data[i] - data[i]);
        sig_sum += std::norm(data[i]);
    }
    lb.max_error = max_err;
    lb.nmse_db = sig_sum > 0 ? 10.0f * std::log10(err_sum / sig_sum) : -999.0f;

    // CSI NMSE for an ideal flat channel. The estimate intentionally includes
    // the transmitter waveform scale and any common complex RF gain, so remove
    // that common gain before measuring frequency-selective error.
    cf32 common_gain{};
    int occupied = 0;
    for (int i = 0; i < FFT_SIZE; ++i) {
        if (std::abs(HT_LTF_F[i]) > 0.1f) {
            common_gain += best.csi[i];
            ++occupied;
        }
    }
    if (occupied) common_gain /= static_cast<float>(occupied);
    double csi_err = 0, csi_sig = 0;
    for (int i = 0; i < FFT_SIZE; i++) {
        if (std::abs(HT_LTF_F[i]) > 0.1f) {
            csi_err += std::norm(best.csi[i] - common_gain);
            csi_sig += std::norm(common_gain);
        }
    }
    lb.csi_nmse_db = csi_sig > 0 ? 10.0f * std::log10(csi_err / csi_sig) : -999.0f;

    return lb;
}

} // namespace ofdm
