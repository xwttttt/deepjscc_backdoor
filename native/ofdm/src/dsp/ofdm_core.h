#pragma once
#include <complex>
#include <cstdint>
#include <cstddef>

using cf32 = std::complex<float>;

namespace ofdm {

// OFDM parameters
constexpr int FFT_SIZE     = 64;
constexpr int CP_LEN       = 16;
constexpr int SYM_SAMP     = FFT_SIZE + CP_LEN;  // 80
constexpr int SAMPLE_RATE  = 20000000;            // 20 Msps

// Subcarrier layout (HT20 style)
constexpr int NUM_DATA_SC  = 52;
constexpr int NUM_PILOT_SC = 4;
constexpr int NUM_USED_SC  = 56;  // data + pilot

// Pilot subcarrier positions in FFT bin indices
constexpr int PILOT_POS[4] = {7, 21, 43, 57};

// Data subcarrier positions in FFT bin indices (excluding pilots, DC, guards)
// These are bins 1..26, 27(excl), 28..32(neg), 38..63(neg mapped)
// Actually: signed -28..-22, -20..-8, -6..-1, +1..+6, +8..+20, +22..+28

// L-STF (short training field) - frequency domain, FFT bin order [0..63]
extern const cf32 L_STF_F[FFT_SIZE];

// L-LTF (long training field) - frequency domain, FFT bin order [0..63]
extern const cf32 L_LTF_F[FFT_SIZE];

// HT-LTF extended - covers all 56 occupied subcarriers, frequency domain
extern const cf32 HT_LTF_F[FFT_SIZE];

// Pilot polarity sequence (127-period)
extern const float PILOT_POLARITY[127];

// Map signed subcarrier index (-32..+31) to FFT bin (0..63)
inline int sc_to_fft(int sc) {
    return (sc + FFT_SIZE) % FFT_SIZE;
}

// Map FFT bin (0..63) to signed subcarrier (-32..+31)
inline int fft_to_sc(int bin) {
    return bin < FFT_SIZE / 2 ? bin : bin - FFT_SIZE;
}

// Ordered list of 52 data subcarrier signed indices, from most negative to most positive
extern const int DATA_SC_ORDER[52];

// Map a 52-index position (0..51) to FFT bin
inline int data_pos_to_fft(int pos) {
    return sc_to_fft(DATA_SC_ORDER[pos]);
}

} // namespace ofdm
