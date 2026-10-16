#include "dsp/preamble.h"
#include <liquid/liquid.h>
#include <cstring>

namespace ofdm {

// IFFT helper: frequency domain -> time domain (unnormalized, matching liquid convention)
static void ifft_64(const cf32* freq, cf32* time) {
    fftplan q = fft_create_plan(64,
        const_cast<liquid_float_complex*>(
            reinterpret_cast<const liquid_float_complex*>(freq)),
        reinterpret_cast<liquid_float_complex*>(time),
        LIQUID_FFT_BACKWARD, 0);
    fft_execute(q);
    fft_destroy_plan(q);
    for (int i = 0; i < FFT_SIZE; ++i) time[i] /= static_cast<float>(FFT_SIZE);
}

void generate_stf(cf32* out) {
    // STF: 10 identical short training symbols, each 16 samples
    // First IFFT the STF frequency domain
    cf32 time_sym[64];
    ifft_64(L_STF_F, time_sym);
    // The STF has periodicity of 16 samples (every 4th subcarrier is active)
    // Copy the first 16 samples 10 times
    for (int i = 0; i < 10; i++) {
        memcpy(&out[i * 16], time_sym, 16 * sizeof(cf32));
    }
}

void generate_ltf(cf32* out) {
    // LTF: long CP (32 samples) + 2 identical long training symbols (128 samples)
    cf32 time_sym[64];
    ifft_64(L_LTF_F, time_sym);
    // CP is the last 32 samples of one symbol (which equals the last 32 of the second)
    // Actually for LTF, the "CP" is the second half of the symbol (32 samples)
    // Then we have 2 full symbols
    // out[0..31] = time_sym[32..63]  (cyclic prefix = last 32 of symbol)
    // out[32..95] = time_sym[0..63]  (first LTF symbol)
    // out[96..159] = time_sym[0..63] (second LTF symbol)
    for (int i = 0; i < 32; i++) {
        out[i] = time_sym[32 + i];
    }
    memcpy(&out[32], time_sym, 64 * sizeof(cf32));
    memcpy(&out[96], time_sym, 64 * sizeof(cf32));
}

void generate_ht_ltf(cf32* out) {
    // HT-LTF: normal CP (16 samples) + 1 HT-LTF symbol (80 samples)
    cf32 time_sym[64];
    ifft_64(HT_LTF_F, time_sym);
    // Use the same 52-tone gain as the legacy L-LTF.  The project block is
    // equalized with the verified legacy CSI, so keeping one IFFT gain across
    // the hybrid frame avoids a hidden 52/56-tone amplitude discontinuity.
    constexpr float ht_scale = static_cast<float>(FFT_SIZE) / 7.211102550927978f;
    for (auto& value : time_sym) value *= ht_scale;
    // CP = last 16 samples
    for (int i = 0; i < CP_LEN; i++) {
        out[i] = time_sym[FFT_SIZE - CP_LEN + i];
    }
    memcpy(&out[CP_LEN], time_sym, FFT_SIZE * sizeof(cf32));
}

} // namespace ofdm
