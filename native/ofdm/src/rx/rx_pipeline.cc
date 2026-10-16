#include "rx/rx_pipeline.h"
#include "dsp/subcarrier_map.h"
#include "dsp/crc16.h"
#include "dsp/preamble.h"
#include "wifi/wifi_anchor.h"
#include <liquid/liquid.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace ofdm {

// Marconi refines the Schmidl-Cox plateau against the known L-LTF before it
// starts the FFT.  The plateau alone can be tens of samples wide, and using
// its left edge as the FFT origin is especially fragile after an SDR sample
// buffer boundary or a short multipath channel.  Search a small neighborhood
// around the STF result and choose the earliest near-maximum LTF match.
static int refine_ltf_timing(const cf32* samples, int num_samples,
                             int approximate_start, float coarse_cfo,
                             float& match_peak) {
    match_peak = 0.0f;
    if (approximate_start < 0 || approximate_start + 192 + 64 > num_samples) return approximate_start;

    cf32 known[160]{};
    generate_ltf(known);
    // The Schmidl-Cox metric can rise before the actual burst when the input
    // starts with zeros/noise. Marconi searches back 24 and forward 112
    // samples from that point, allowing the LTF to determine the FFT origin.
    const int first = std::max(0, approximate_start - 24);
    const int last = std::min(num_samples - 256, approximate_start + 112);
    if (first > last) return approximate_start;

    struct Candidate { int start; float score; };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(last - first + 1));
    double reference_power = 0.0;
    for (int i = 0; i < 64; ++i) reference_power += std::norm(known[32 + i]);

    int best_start = approximate_start;
    for (int start = first; start <= last; ++start) {
        const int ltf_fft = start + 192;
        cf32 correlation(0.0f, 0.0f);
        double input_power = 0.0;
        for (int i = 0; i < 64; ++i) {
            const int index = ltf_fft + i;
            const float phase = static_cast<float>(index) * coarse_cfo;
            const cf32 corrected = samples[index] * cf32(std::cos(phase), std::sin(phase));
            correlation += corrected * std::conj(known[32 + i]);
            input_power += std::norm(corrected);
        }
        const float denominator = static_cast<float>(std::sqrt(std::max(input_power * reference_power, 1.0e-20)));
        const float score = denominator > 0.0f ? std::abs(correlation) / denominator : 0.0f;
        candidates.push_back({start, score});
        if (score > match_peak) {
            match_peak = score;
            best_start = start;
        }
    }

    // Keep an early point on the correlation lobe, as Marconi does, instead
    // of locking to a noisy right-hand maximum.
    // Marconi chooses the earliest point on the L-LTF correlation lobe only
    // when it is a strong normalized match; otherwise it keeps the global
    // maximum.  Do not accept a weak 0.2 noise match as a timing lock.
    const float early_threshold = std::max(0.72f, match_peak * 0.97f);
    for (const auto& candidate : candidates) {
        if (candidate.score >= early_threshold) {
            best_start = candidate.start;
            break;
        }
    }
    return best_start;
}

RxPipeline::RxPipeline() {
    buffer_.reserve(262144);
}

void RxPipeline::reset() {
    buffer_.clear();
    scan_offset_ = 0;
    dc_ = cf32(0.0f, 0.0f);
}

void RxPipeline::fft_64(const cf32* in, cf32* out) {
    fftplan q = fft_create_plan(64,
        const_cast<liquid_float_complex*>(
            reinterpret_cast<const liquid_float_complex*>(in)),
        reinterpret_cast<liquid_float_complex*>(out),
        LIQUID_FFT_FORWARD, 0);
    fft_execute(q);
    fft_destroy_plan(q);
}

void RxPipeline::ifft_64(const cf32* in, cf32* out) {
    fftplan q = fft_create_plan(64,
        const_cast<liquid_float_complex*>(
            reinterpret_cast<const liquid_float_complex*>(in)),
        reinterpret_cast<liquid_float_complex*>(out),
        LIQUID_FFT_BACKWARD, 0);
    fft_execute(q);
    fft_destroy_plan(q);
}

void RxPipeline::compensate_cfo(cf32* samples, int num, int sample_offset, float cfo_rad) {
    for (int i = 0; i < num; i++) {
        float phase = (float)(sample_offset + i) * cfo_rad;
        samples[i] *= cf32(std::cos(phase), std::sin(phase));
    }
}

// STF autocorrelation detection (adapted from gr-ieee80211 trigger_impl.cc and sync_impl.cc)
// Sliding window: compute |R(d)| / sqrt(P1*P2) where R(d) = sum r(n)*conj(r(n+16))
int RxPipeline::detect_stf(const cf32* samples, int num_samples, float& ac_peak) {
    if (num_samples < 80) return -1;

    const int LAG = 16;  // STF periodicity
    const int WIN = 64;  // correlation window (4 short symbols)
    const int SEARCH_LEN = num_samples - WIN - LAG;

    if (SEARCH_LEN < 100) return -1;

    // Sliding autocorrelation
    // R(d) = sum_{n=0}^{WIN-1} r(d+n) * conj(r(d+n+LAG))
    cf32 r_sum(0, 0);
    float p1_sum = 0, p2_sum = 0;

    // Initialize
    for (int n = 0; n < WIN; n++) {
        r_sum += samples[n] * std::conj(samples[n + LAG]);
        p1_sum += std::norm(samples[n]);
        p2_sum += std::norm(samples[n + LAG]);
    }

    std::vector<float> ac(SEARCH_LEN);
    for (int d = 0; d < SEARCH_LEN; d++) {
        float denom = std::sqrt(p1_sum) * std::sqrt(p2_sum);
        ac[d] = denom > 1e-10f ? std::abs(r_sum) / denom : 0.0f;

        // Slide
        if (d + 1 < SEARCH_LEN) {
            r_sum -= samples[d] * std::conj(samples[d + LAG]);
            p1_sum -= std::norm(samples[d]);
            p2_sum -= std::norm(samples[d + LAG]);
            r_sum += samples[d + WIN] * std::conj(samples[d + WIN + LAG]);
            p1_sum += std::norm(samples[d + WIN]);
            p2_sum += std::norm(samples[d + WIN + LAG]);
        }
    }

    // Marconi's native Wi-Fi service uses a fixed 0.58 Schmidl-Cox gate and
    // lets the known L-LTF validation reject false candidates.  A threshold
    // derived from the largest correlation in the whole block is brittle:
    // one periodic interferer can raise it above the real STF plateau.
    ac_peak = 0;
    for (int d = 0; d < SEARCH_LEN; d++) {
        if (ac[d] > ac_peak) {
            ac_peak = ac[d];
        }
    }

    if (ac_peak < 0.5f) return -1;

    // Use the earliest sustained high-correlation plateau. Searching for the
    // global peak can lock onto an accidentally periodic payload later in the
    // buffer, especially when several bursts are processed together.
    // Match Marconi's service-side legacy Wi-Fi default (0.58). A low fixed threshold
    // causes noise/DC plateaus to trigger the expensive LTF/FFT path.
    constexpr float threshold = 0.58f;
    int plateau_start = -1;
    for (int d = 0; d < SEARCH_LEN; ++d) {
        if (ac[d] >= threshold) { plateau_start = d; break; }
    }
    if (plateau_start < 0) return -1;
    // With the detector's 64-sample window, the first sustained point is the
    // burst start. Returning it avoids deriving timing from the right edge of
    // a plateau, which is distorted by the following LTF and payload.
    return plateau_start;
}

// Coarse CFO from STF (adapted from sync_impl.cc: ltf_cfo)
// Uses 16-sample periodicity: angle(r(n) * conj(r(n+16))) / 16
float RxPipeline::estimate_cfo_coarse(const cf32* samples, int stf_end_idx) {
    // Average over a few STF periods
    cf32 sum(0, 0);
    int num_avg = 4;  // average over 4 periods
    // The delayed term reaches 16 samples beyond the leading term. Keep
    // both sides inside the STF; otherwise the window leaks into the L-LTF
    // CP and biases CFO when the capture has a non-zero start offset.
    int start = std::max(0, stf_end_idx - (num_avg + 1) * 16);
    for (int i = 0; i < num_avg * 16; i++) {
        sum += samples[start + i] * std::conj(samples[start + i + 16]);
    }
    return std::arg(sum / (float)(num_avg * 16)) / 16.0f;
}

// Fine CFO from LTF cross-correlation (adapted from sync_impl.cc: ltf_cfo)
// Uses 64-sample periodicity in LTF
// LTF layout: [CP(32)] [sym1(64)] [sym2(64)]
float RxPipeline::estimate_cfo_fine(const cf32* samples, int ltf_start_idx, float coarse_cfo) {
    // Compensate coarse CFO and extract the two LTF symbols (skip CP=32)
    cf32 sym1[64], sym2[64];
    for (int i = 0; i < 64; i++) {
        int idx1 = ltf_start_idx + 32 + i;
        int idx2 = ltf_start_idx + 96 + i;
        float p1 = (float)idx1 * coarse_cfo;
        float p2 = (float)idx2 * coarse_cfo;
        sym1[i] = samples[idx1] * cf32(std::cos(p1), std::sin(p1));
        sym2[i] = samples[idx2] * cf32(std::cos(p2), std::sin(p2));
    }

    // Cross-correlate the two LTF symbols
    cf32 sum(0, 0);
    for (int i = 0; i < 64; i++) {
        sum += sym1[i] * std::conj(sym2[i]);
    }
    float fine_cfo = std::arg(sum / 64.0f) / 64.0f;
    return coarse_cfo + fine_cfo;
}

// Channel estimation follows Marconi's legacy path: average the two repeated
// L-LTF symbols wherever the standard sequence is defined, and use the
// project HT-LTF only for the two extra edge subcarriers (+/-27,+/-28).
void RxPipeline::estimate_channel(const cf32* samples, int ltf_start_idx,
                                   int hltf_start_idx, float cfo_rad, cf32* csi,
                                   bool prefer_ht) {
    cf32 ltf1[64], ltf2[64], ht_time[64], ht_freq[64];
    for (int i = 0; i < 64; ++i) {
        const int ltf1_idx = ltf_start_idx + 32 + i;
        const int ltf2_idx = ltf_start_idx + 96 + i;
        const int ht_idx = hltf_start_idx + CP_LEN + i;
        const float ltf1_phase = static_cast<float>(ltf1_idx) * cfo_rad;
        const float ltf2_phase = static_cast<float>(ltf2_idx) * cfo_rad;
        const float ht_phase = static_cast<float>(ht_idx) * cfo_rad;
        ltf1[i] = samples[ltf1_idx] * cf32(std::cos(ltf1_phase), std::sin(ltf1_phase));
        ltf2[i] = samples[ltf2_idx] * cf32(std::cos(ltf2_phase), std::sin(ltf2_phase));
        ht_time[i] = samples[ht_idx] * cf32(std::cos(ht_phase), std::sin(ht_phase));
    }
    cf32 ltf1_freq[64], ltf2_freq[64];
    fft_64(ltf1, ltf1_freq);
    fft_64(ltf2, ltf2_freq);
    fft_64(ht_time, ht_freq);

    for (int i = 0; i < FFT_SIZE; ++i) {
        // New extension bursts use the HT-LTF immediately preceding their
        // payload for all 56 occupied bins. The legacy-compatible path keeps
        // L-LTF authoritative and only obtains its edge bins from HT-LTF.
        if (prefer_ht && std::abs(HT_LTF_F[i]) > 0.1f) {
            csi[i] = ht_freq[i] / HT_LTF_F[i];
        } else if (std::abs(L_LTF_F[i]) > 0.1f) {
            csi[i] = (ltf1_freq[i] + ltf2_freq[i]) * 0.5f / L_LTF_F[i];
        } else if (std::abs(HT_LTF_F[i]) > 0.1f) {
            csi[i] = ht_freq[i] / HT_LTF_F[i];
        } else {
            csi[i] = cf32(0, 0);
        }
    }
}

// Noise variance from LTF difference
float RxPipeline::estimate_noise_var(const cf32* samples, int ltf_start_idx,
                                      float cfo_rad, const cf32* csi) {
    (void)csi;
    // Extract two LTF symbols (after the 32-sample CP)
    cf32 sym1[64], sym2[64];
    for (int i = 0; i < 64; i++) {
        int idx1 = ltf_start_idx + 32 + i;
        int idx2 = ltf_start_idx + 96 + i;
        float p1 = (float)idx1 * cfo_rad;
        float p2 = (float)idx2 * cfo_rad;
        sym1[i] = samples[idx1] * cf32(std::cos(p1), std::sin(p1));
        sym2[i] = samples[idx2] * cf32(std::cos(p2), std::sin(p2));
    }

    cf32 f1[64], f2[64];
    fft_64(sym1, f1);
    fft_64(sym2, f2);

    // Two identical LTF observations contain independent noise. Their
    // difference therefore has twice the per-observation noise variance.
    float noise_power = 0.0f;
    int count = 0;
    for (int i = 0; i < 64; i++) {
        if (std::abs(L_LTF_F[i]) > 0.1f) {
            noise_power += 0.5f * std::norm(f1[i] - f2[i]);
            ++count;
        }
    }
    return std::max(noise_power / std::max(1, count), 1e-12f);
}

// MMSE equalization (from demod_impl.cc: nonLegacyChanUpdate, adapted)
void RxPipeline::equalize_symbol(const cf32* time_in, float cfo_rad, const cf32* csi,
                                   int pilot_idx, float noise_var, cf32* eq_out) {
    // CFO compensation + FFT
    cf32 time_comp[64];
    for (int i = 0; i < 64; i++) {
        float phase = (float)i * cfo_rad;
        time_comp[i] = time_in[i] * cf32(std::cos(phase), std::sin(phase));
    }

    cf32 freq[64];
    fft_64(time_comp, freq);

    // MMSE equalization: H_eq = conj(H) / (|H|^2 + sigma^2)
    for (int i = 0; i < FFT_SIZE; i++) {
        float h2 = std::norm(csi[i]);
        if (h2 > 1e-10f) {
            if (use_mmse_ && noise_var > 0) {
                eq_out[i] = freq[i] * std::conj(csi[i]) / (h2 + noise_var);
            } else {
                // ZF
                eq_out[i] = freq[i] / csi[i];
            }
        } else {
            eq_out[i] = cf32(0, 0);
        }
    }

    // Pilot-based phase tracking (from demod_impl.cc: nonLegacyChanUpdate)
    // Extract pilots and compute common phase
    const auto track = extract_pilots_full(eq_out, pilot_idx);
    for (int bin = 0; bin < FFT_SIZE; ++bin) {
        const int sc = fft_to_sc(bin);
        const float correction = -(track.common_phase + track.phase_slope * static_cast<float>(sc));
        eq_out[bin] *= std::polar(1.0f, correction);
    }
}

void RxPipeline::equalize_symbol_zf(const cf32* time_in, float cfo_rad, const cf32* csi,
                                      int pilot_idx, cf32* eq_out) {
    equalize_symbol(time_in, cfo_rad, csi, pilot_idx, 0.0f, eq_out);
}

std::vector<RxResult> RxPipeline::process(const cf32* samples, int num_samples) {
    if (!samples && num_samples > 0) throw std::invalid_argument("OFDM RX input must not be null");
    if (num_samples > 0) {
        // Marconi's conditioner removes the slowly varying DC component
        // before synchronization.  B210 captures commonly contain a sizable
        // LO/DC term; leaving it in the Schmidl-Cox power estimate creates
        // false plateaus and wastes decoder work.
        constexpr float dc_alpha = 0.00001f;
        const auto old_size = buffer_.size();
        buffer_.resize(old_size + static_cast<std::size_t>(num_samples));
        for (int index = 0; index < num_samples; ++index) {
            dc_ = dc_ * (1.0f - dc_alpha) + samples[index] * dc_alpha;
            buffer_[old_size + static_cast<std::size_t>(index)] = samples[index] - dc_;
        }
    }
    std::vector<RxResult> results;

    // First acquire the complete standard legacy Wi-Fi control anchor.
    // Do not require the following project OFDM block to be present before
    // attempting the anchor: in streaming mode it may arrive in a later USB
    // callback, and the old LTF-relative minimum dropped otherwise-valid
    // short captures before synchronization could even be counted.
    const int MIN_BURST_LEN = wifi_anchor::kFrameSamples;

    int search_start = static_cast<int>(std::min(scan_offset_, buffer_.size()));
    const auto input_size = static_cast<int>(buffer_.size());
    const auto input = buffer_.data();
    while (search_start + MIN_BURST_LEN < input_size) {
        // Step 1: Detect STF
        float ac_peak = 0;
        int stf_sync = detect_stf(input + search_start, input_size - search_start, ac_peak);
        if (stf_sync < 0) {
            // Marconi advances its sliding scan offset after a failed
            // candidate instead of rescanning the entire capture on every
            // callback. Keep a tail for a preamble crossing the next block.
            search_start = std::max(0, input_size - 1024);
            break;
        }
        ++diagnostics_.stf_candidates;

        int burst_start = search_start + stf_sync;
        int ltf_start = burst_start + 160;

        // Need the complete standard anchor.  The project HT-LTF/data tail is
        // checked separately after the anchor metadata has been validated.
        if (burst_start + MIN_BURST_LEN > input_size) break;

        // Step 2: Estimate CFO from STF
        // STF ends at ltf_start, use samples just before that for coarse CFO
        float cfo_coarse = 0;
        if (ltf_start > 16) {
            cfo_coarse = estimate_cfo_coarse(input, ltf_start);
        }

        float ltf_match = 0.0f;
        const int refined_start = refine_ltf_timing(input, input_size, burst_start,
                                                     cfo_coarse, ltf_match);
        if (refined_start >= 0 && ltf_match >= 0.20f) {
            ++diagnostics_.ltf_locks;
            burst_start = refined_start;
            ltf_start = burst_start + 160;
            // Recompute the coarse CFO using the refined STF/LTF boundary.
            if (ltf_start > 16) cfo_coarse = estimate_cfo_coarse(input, ltf_start);
        }

        // First try the standard Wi-Fi anchor. A successful version-2 anchor
        // means the following samples are project raw-I/Q symbols, so no
        // custom BPSK control header is consumed here. The decoder below keeps
        // compatibility with the original pre-anchor control-symbol format.
        if (burst_start >= 0 && static_cast<std::size_t>(burst_start) + wifi_anchor::kFrameSamples <=
            static_cast<std::size_t>(input_size)) {
            const auto anchor = wifi_anchor::decode(input, input_size,
                                                    static_cast<std::size_t>(burst_start));
            wifi_anchor::ExtensionControl extension;
            if (anchor.ok && wifi_anchor::decode_extension_control(anchor.metadata, extension)) {
                const std::size_t original_length = extension.original_length;
                const int num_data_syms = extension.num_data_symbols;
                const int hltf_start = burst_start + static_cast<int>(wifi_anchor::kFrameSamples + wifi_anchor::kGuardSamples);
                const int data_start = hltf_start + 80;
                if (original_length == 0 || num_data_syms <= 0 ||
                    data_start + num_data_syms * SYM_SAMP > input_size) {
                    break;
                }

                // Estimate all 56 occupied extension bins from the HT-LTF
                // immediately preceding the raw complex OFDM blocks.
                cf32 csi[FFT_SIZE];
                estimate_channel(input, ltf_start, hltf_start, anchor.cfo_rad, csi, true);
                const float noise_var = estimate_noise_var(input, ltf_start, anchor.cfo_rad, csi);
                float sig_power = 0.0f;
                int occupied_bins = 0;
                for (int bin = 0; bin < FFT_SIZE; ++bin) {
                    if (std::abs(HT_LTF_F[bin]) > 0.1f) {
                        sig_power += std::norm(csi[bin]);
                        ++occupied_bins;
                    }
                }
                sig_power /= static_cast<float>(std::max(1, occupied_bins));
                const float csi_quality_db = 10.0f * std::log10(std::max(1e-12f, sig_power / std::max(noise_var, 1e-12f)));
                diagnostics_.last_csi_quality_db = csi_quality_db;
                diagnostics_.max_csi_quality_db = std::max(diagnostics_.max_csi_quality_db, csi_quality_db);
                if (!std::isfinite(csi_quality_db) || csi_quality_db < 3.0f) {
                    ++diagnostics_.csi_rejects;
                    search_start = burst_start + static_cast<int>(wifi_anchor::kFrameSamples);
                    continue;
                }

                std::vector<cf32> recovered;
                recovered.reserve(static_cast<std::size_t>(num_data_syms * NUM_DATA_SC));
                for (int sym = 0; sym < num_data_syms; ++sym) {
                    cf32 eq[FFT_SIZE];
                    equalize_symbol(&input[data_start + sym * SYM_SAMP + CP_LEN],
                                   anchor.cfo_rad, csi, sym, noise_var, eq);
                    cf32 data_sym[NUM_DATA_SC];
                    map_freq_to_data(eq, data_sym);
                    recovered.insert(recovered.end(), data_sym, data_sym + NUM_DATA_SC);
                }
                recovered.resize(original_length);

                RxResult result;
                result.success = true;
                result.meta.burst_id = extension.burst_id;
                result.meta.original_length = static_cast<std::uint32_t>(original_length);
                result.meta.num_data_symbols = num_data_syms;
                result.meta.cfo_hz = anchor.cfo_hz;
                result.meta.rssi = sig_power;
                result.meta.snr_db = csi_quality_db;
                result.meta.csi_quality_db = csi_quality_db;
                result.meta.noise_var = noise_var;
                result.meta.crc_ok = true;
                result.meta.sync_offset = ltf_start;
                std::memcpy(result.csi, csi, sizeof(csi));
                result.data = std::move(recovered);
                results.push_back(std::move(result));
                ++diagnostics_.decoded;
                search_start = data_start + num_data_syms * SYM_SAMP + CP_LEN;
                continue;
            }
        }

        // Step 3: Fine CFO from LTF
        // LTF starts at ltf_start (includes 32-sample CP)
        if (ltf_start + 128 >= input_size) break;
        float cfo_rad = estimate_cfo_fine(input, ltf_start, cfo_coarse);
        float cfo_hz = cfo_rad * (float)SAMPLE_RATE / (2.0f * M_PI);
        diagnostics_.last_cfo_hz = cfo_hz;

        // Step 4: Channel estimation from HT-LTF
        // HT-LTF starts after LTF (160 samples): ltf_start + 160
        int hltf_start = ltf_start + 160;
        if (hltf_start + 80 >= input_size) break;

        cf32 csi[FFT_SIZE];
        estimate_channel(input, ltf_start, hltf_start, cfo_rad, csi);

        // Step 5: Noise variance estimation
        float noise_var = estimate_noise_var(input, ltf_start, cfo_rad, csi);

        // Compute channel power only on occupied HT-LTF bins.  A CRC hit by
        // itself is not enough to publish CSI: in RF noise a false STF/CRC
        // lock can otherwise produce a plausible-looking random channel.
        float sig_power = 0.0f;
        int occupied_bins = 0;
        for (int i = 0; i < FFT_SIZE; i++) {
            if (std::abs(HT_LTF_F[i]) > 0.1f) {
                if (!std::isfinite(csi[i].real()) || !std::isfinite(csi[i].imag())) {
                    sig_power = 0.0f;
                    occupied_bins = 0;
                    break;
                }
                sig_power += std::norm(csi[i]);
                ++occupied_bins;
            }
        }
        if (occupied_bins == 0 || !std::isfinite(noise_var) || noise_var <= 0.0f) {
            ++diagnostics_.csi_rejects;
            search_start = ltf_start + 160;
            continue;
        }
        sig_power /= static_cast<float>(occupied_bins);
        const float csi_quality_db = 10.0f * std::log10(std::max(1e-12f, sig_power / noise_var));
        diagnostics_.last_csi_quality_db = csi_quality_db;
        diagnostics_.max_csi_quality_db = std::max(diagnostics_.max_csi_quality_db, csi_quality_db);
        // Keep a conservative 3 dB gate.  This rejects noise-only CSI while
        // still allowing low-SNR but decodable packets through.
        if (!std::isfinite(csi_quality_db) || csi_quality_db < 3.0f) {
            ++diagnostics_.csi_rejects;
            search_start = ltf_start + 160;
            continue;
        }
        const float rssi = sig_power;

        // Step 6: Decode control header
        // 3 control symbols start after HT-LTF: ltf_start + 160 + 80 = ltf_start + 240
        int ctrl_start = ltf_start + 240;
        if (ctrl_start + 3 * SYM_SAMP >= input_size) break;

        cf32 ctrl_freq[CTRL_HDR_NUM_SYMBOLS * FFT_SIZE];
        for (int sym = 0; sym < CTRL_HDR_NUM_SYMBOLS; sym++) {
            int sym_start = ctrl_start + sym * SYM_SAMP;
            // Skip CP (16 samples), take 64 samples with CFO comp
            cf32 time_sym[64];
            for (int i = 0; i < 64; i++) {
                int idx = sym_start + CP_LEN + i;
                float phase = (float)idx * cfo_rad;
                time_sym[i] = input[idx] * cf32(std::cos(phase), std::sin(phase));
            }
            fft_64(time_sym, &ctrl_freq[sym * FFT_SIZE]);
        }

        uint8_t ctrl_bits[CTRL_HDR_TOTAL_BITS] = {};
        decode_ctrl_symbols(ctrl_freq, csi, ctrl_bits, 0);

        ControlHeader hdr;
        bool crc_ok = decode_control_header(ctrl_bits, hdr);

        if (!crc_ok || hdr.magic != CTRL_HDR_MAGIC) {
            ++diagnostics_.control_failures;
            // Failed CRC, try next
            search_start = ltf_start + 160;
            continue;
        }

        const auto expected_symbols = (hdr.original_length + NUM_DATA_SC - 1U) / NUM_DATA_SC;
        if (hdr.original_length == 0 || hdr.num_data_symbols == 0 ||
            hdr.num_data_symbols != expected_symbols) {
            ++diagnostics_.header_failures;
            search_start = ltf_start + 160;
            continue;
        }

        // Step 7: Decode data symbols
        int num_data_syms = hdr.num_data_symbols;
        int data_start = ctrl_start + CTRL_HDR_NUM_SYMBOLS * SYM_SAMP;

        if (data_start + num_data_syms * SYM_SAMP > input_size) {
            search_start = ltf_start + 160;
            continue;
        }

        // Pilot polarity for data starts after control header
        int pilot_start = CTRL_HDR_NUM_SYMBOLS;

        // SNR estimate from control header (BPSK) residual
        float snr_db = 10.0f * std::log10(std::max(1e-10f, sig_power / std::max(noise_var, 1e-10f)));

        std::vector<cf32> recovered;
        recovered.reserve(num_data_syms * NUM_DATA_SC);

        for (int sym = 0; sym < num_data_syms; sym++) {
            int sym_start = data_start + sym * SYM_SAMP;

            // Equalize with CFO compensation, MMSE, pilot tracking
            cf32 eq[64];
            if (use_mmse_) {
                equalize_symbol(&input[sym_start + CP_LEN], cfo_rad, csi,
                                pilot_start + sym, noise_var, eq);
            } else {
                equalize_symbol_zf(&input[sym_start + CP_LEN], cfo_rad, csi,
                                   pilot_start + sym, eq);
            }

            // Extract 52 data subcarriers
            cf32 data_sym[NUM_DATA_SC];
            map_freq_to_data(eq, data_sym);

            for (int i = 0; i < NUM_DATA_SC; i++) {
                recovered.push_back(data_sym[i]);
            }
        }

        // Trim to original length
        if ((int)recovered.size() > (int)hdr.original_length) {
            recovered.resize(hdr.original_length);
        }

        RxResult result;
        result.success = true;
        result.meta.burst_id = hdr.burst_id;
        result.meta.original_length = hdr.original_length;
        result.meta.num_data_symbols = hdr.num_data_symbols;
        result.meta.cfo_hz = cfo_hz;
        result.meta.rssi = rssi;
        result.meta.snr_db = snr_db;
        result.meta.csi_quality_db = csi_quality_db;
        result.meta.noise_var = noise_var;
        result.meta.crc_ok = crc_ok;
        result.meta.sync_offset = ltf_start;
        memcpy(result.csi, csi, sizeof(csi));
        result.data = std::move(recovered);
        results.push_back(std::move(result));
        ++diagnostics_.decoded;

        // Move past this burst
        search_start = data_start + num_data_syms * SYM_SAMP + CP_LEN;
    }

    scan_offset_ = static_cast<std::size_t>(std::max(0, search_start));
    if (scan_offset_ > 2048) {
        const auto discard = scan_offset_ - 512;
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(discard));
        scan_offset_ -= discard;
    } else if (buffer_.size() > 14 * 1024 * 1024) {
        const auto discard = buffer_.size() - 8192;
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(discard));
        scan_offset_ = 0;
    }
    return results;
}

} // namespace ofdm
