#include "tx/tx_pipeline.h"
#include "dsp/preamble.h"
#include "dsp/subcarrier_map.h"
#include "wifi/wifi_anchor.h"
#include <liquid/liquid.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

namespace ofdm {

int TxPipeline::calc_num_data_symbols(size_t len) {
    return (int)((len + NUM_DATA_SC - 1) / NUM_DATA_SC);
}

int TxPipeline::calc_burst_samples(int num_data_symbols) {
    // Standard legacy Wi-Fi anchor + guard + project HT-LTF + raw-I/Q
    // symbols. The anchor size is derived from its robust control payload.
    return static_cast<int>(wifi_anchor::kFrameSamples) +
           static_cast<int>(wifi_anchor::kGuardSamples) + 80 +
           num_data_symbols * SYM_SAMP + CP_LEN;
}

TxPipeline::Result TxPipeline::build_burst(const cf32* data, size_t len, uint32_t burst_id) {
    if (data == nullptr || len == 0) throw std::invalid_argument("OFDM burst data must not be empty");
    if (len > std::numeric_limits<uint32_t>::max()) throw std::length_error("OFDM burst exceeds header length field");
    Result result;
    int num_data_syms = calc_num_data_symbols(len);
    if (num_data_syms > std::numeric_limits<uint16_t>::max()) throw std::length_error("OFDM burst has too many data symbols");
    int total_samples = calc_burst_samples(num_data_syms);
    result.samples.reserve(total_samples);

    // Keep the old project header populated for API/telemetry compatibility;
    // the on-air header is the standards-compatible Wi-Fi anchor below.
    ControlHeader& hdr = result.header;
    hdr.magic = CTRL_HDR_MAGIC;
    hdr.version_flags = 0;
    hdr.burst_id = burst_id;
    hdr.original_length = (uint32_t)len;
    hdr.num_data_symbols = (uint16_t)num_data_syms;

    const auto metadata = wifi_anchor::encode_extension_control({
        burst_id,
        static_cast<std::uint32_t>(len),
        static_cast<std::uint16_t>(num_data_syms),
    });
    const auto anchor = wifi_anchor::build(metadata);
    result.samples.insert(result.samples.end(), anchor.begin(), anchor.end());
    result.samples.insert(result.samples.end(), wifi_anchor::kGuardSamples, cf32{});

    // Project HT-LTF extends the legacy channel estimate to ±28 and is used
    // only for the raw complex OFDM payload. Guard/DC bins remain zero.
    const int hltf_offset = static_cast<int>(result.samples.size());
    result.samples.resize(result.samples.size() + 80);
    generate_ht_ltf(&result.samples[hltf_offset]);
    const int offset = static_cast<int>(result.samples.size());

    // Data symbols: raw CF32 passthrough (52 values per symbol).  The pilot
    // sequence starts at zero for the first project payload symbol, matching
    // the HT/Marconi convention after a fresh training field.
    result.samples.resize(result.samples.size() + static_cast<std::size_t>(num_data_syms * SYM_SAMP));
    encode_data_symbols(data, len, num_data_syms, &result.samples[offset], 0);
    result.samples.insert(result.samples.end(), CP_LEN, cf32{});

    // Keep the complete waveform inside a conservative CF32 full-scale range.
    // Applying one common factor preserves the training/data relationship, and
    // the receiver removes it as part of the channel estimate.
    float peak = 0.0f;
    for (const auto& sample : result.samples) peak = std::max(peak, std::abs(sample));
    if (peak > 0.8f) {
        result.waveform_scale = 0.8f / peak;
        for (auto& sample : result.samples) sample *= result.waveform_scale;
    }

    return result;
}

void TxPipeline::encode_data_symbols(const cf32* data, size_t len, int num_syms,
                                      cf32* out, int pilot_start) {
    for (int sym = 0; sym < num_syms; sym++) {
        cf32 freq[64] = {};

        // Fill data subcarriers
        int data_offset = sym * NUM_DATA_SC;
        for (int i = 0; i < NUM_DATA_SC; i++) {
            int idx = data_offset + i;
            if (idx < (int)len) {
                int bin = sc_to_fft(DATA_SC_ORDER[i]);
                freq[bin] = data[idx];
            }
        }

        // Insert pilots
        insert_pilots(freq, pilot_start + sym);

        // IFFT (unnormalized, matching the legacy Wi-Fi anchor).  Scale by
        // the 52 legacy occupied tones so raw blocks have the same complex
        // gain as the L-LTF-derived CSI used by the receiver.
        cf32 time[64];
        fftplan q = fft_create_plan(64,
            reinterpret_cast<liquid_float_complex*>(freq),
            reinterpret_cast<liquid_float_complex*>(time),
            LIQUID_FFT_BACKWARD, 0);
        fft_execute(q);
        fft_destroy_plan(q);
        constexpr float active_tone_scale = 1.0f / 7.211102550927978f;
        for (auto& sample : time) sample *= active_tone_scale;

        // Add CP
        cf32* sym_out = out + sym * SYM_SAMP;
        for (int i = 0; i < CP_LEN; i++) {
            sym_out[i] = time[FFT_SIZE - CP_LEN + i];
        }
        memcpy(&sym_out[CP_LEN], time, FFT_SIZE * sizeof(cf32));
    }
}

} // namespace ofdm
