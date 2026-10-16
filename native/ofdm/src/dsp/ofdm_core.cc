#include "dsp/ofdm_core.h"
#include <cmath>

namespace ofdm {

// Signed subcarrier index order: -28..-22, -20..-8, -6..-1, +1..+6, +8..+20, +22..+28
// Excluding pilots at ±7, ±21 and DC at 0
const int DATA_SC_ORDER[52] = {
    -28,-27,-26,-25,-24,-23,-22,
    -20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,
    -6,-5,-4,-3,-2,-1,
    +1,+2,+3,+4,+5,+6,
    +8,+9,+10,+11,+12,+13,+14,+15,+16,+17,+18,+19,+20,
    +22,+23,+24,+25,+26,+27,+28
};

// L-STF: sparse frequency domain. Non-zero at subcarriers ±{4,8,12,16,20,24}
// Following IEEE 802.11 standard: S_{-26,26} = sqrt(13/6) * {0,0,1+j,0,...}
// Exact legacy 802.11 STF values, matching Marconi's Wi-Fi PHY.  The
// diagonal +/- (1+j)/sqrt(2) tones are part of the standard preamble; an
// imaginary-only table can still look periodic but is not interoperable.
const cf32 L_STF_F[FFT_SIZE] = {
    // bin 0 (DC)
    cf32(0,0),
    // bins 1-7
    cf32(0,0), cf32(0,0), cf32(0,0),
    cf32(-0.70710678f,-0.70710678f),  // bin 4 = +4
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 8-11
    cf32(-0.70710678f,-0.70710678f),  // bin 8 = +8
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 12-15
    cf32(0.70710678f,0.70710678f),  // bin 12 = +12
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 16-19
    cf32(0.70710678f,0.70710678f),  // bin 16 = +16
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 20-23
    cf32(0.70710678f,0.70710678f),  // bin 20 = +20
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 24-27
    cf32(0.70710678f,0.70710678f),  // bin 24 = +24
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 28-31
    cf32(0,0),  // bin 28 (sc +28 is guard for legacy STF)
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 32-35 (negative freq mapped: -32..-29)
    cf32(0,0), cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 36-39 (subcarrier -28..-25)
    cf32(0,0),  // bin 36 (sc -28 is guard for legacy STF)
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 40-43
    cf32(0.70710678f,0.70710678f),  // bin 40 = sc -24
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 44-47
    cf32(-0.70710678f,-0.70710678f),  // bin 44 = sc -20
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 48-51
    cf32(0.70710678f,0.70710678f),  // bin 48 = sc -16
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 52-55
    cf32(-0.70710678f,-0.70710678f),  // bin 52 = sc -12
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 56-59
    cf32(-0.70710678f,-0.70710678f),  // bin 56 = sc -8
    cf32(0,0), cf32(0,0), cf32(0,0),
    // bins 60-63
    cf32(0.70710678f,0.70710678f), cf32(0,0), cf32(0,0), cf32(0,0),
};

// L-LTF: covers subcarriers ±1..±26 (52 subcarriers)
// Standard IEEE 802.11 LTF sequence
// From gr-ieee80211 C8P_LTF_L_F
const cf32 L_LTF_F[FFT_SIZE] = {
    // FFT bins 0..31: DC, +1..+26, upper guard
    cf32(0,0), cf32(1,0), cf32(-1,0), cf32(-1,0),
    cf32(1,0), cf32(1,0), cf32(-1,0), cf32(1,0),
    cf32(-1,0), cf32(1,0), cf32(-1,0), cf32(-1,0),
    cf32(-1,0), cf32(-1,0), cf32(-1,0), cf32(1,0),
    cf32(1,0), cf32(-1,0), cf32(-1,0), cf32(1,0),
    cf32(-1,0), cf32(1,0), cf32(-1,0), cf32(1,0),
    cf32(1,0), cf32(1,0), cf32(1,0), cf32(0,0),
    cf32(0,0), cf32(0,0), cf32(0,0), cf32(0,0),
    // FFT bins 32..63: lower guard, -26..-1
    cf32(0,0), cf32(0,0), cf32(0,0), cf32(0,0),
    cf32(0,0), cf32(0,0), cf32(1,0), cf32(1,0),
    cf32(-1,0), cf32(-1,0), cf32(1,0), cf32(1,0),
    cf32(-1,0), cf32(1,0), cf32(-1,0), cf32(1,0),
    cf32(1,0), cf32(1,0), cf32(1,0), cf32(1,0),
    cf32(1,0), cf32(-1,0), cf32(-1,0), cf32(1,0),
    cf32(1,0), cf32(-1,0), cf32(1,0), cf32(-1,0),
    cf32(1,0), cf32(1,0), cf32(1,0), cf32(1,0),
};

// Project HT-LTF: covers the 56 occupied subcarriers (±1..±28), including
// pilots at ±7 and ±21.  DC, Nyquist, and guard bins remain zero.
const cf32 HT_LTF_F[FFT_SIZE] = {
    cf32(0,0),  cf32(1,0),  cf32(-1,0), cf32(1,0),  // bins 0..3
    cf32(1,0),  cf32(1,0),  cf32(1,0),  cf32(1,0),  // bins 4..7: sc +4..+7
    cf32(-1,0), cf32(-1,0), cf32(1,0),  cf32(1,0),  // bins 8..11
    cf32(-1,0), cf32(1,0),  cf32(-1,0), cf32(1,0),  // bins 12..15
    cf32(1,0),  cf32(1,0),  cf32(1,0),  cf32(1,0),  // bins 16..19
    cf32(1,0),  cf32(-1,0), cf32(-1,0), cf32(1,0),  // bins 20..23
    cf32(1,0),  cf32(-1,0), cf32(1,0),  cf32(-1,0), // bins 24..27: non-legacy LTF
    cf32(1,0),  cf32(0,0),  cf32(0,0),  cf32(0,0),  // bins 28..31
    cf32(0,0),  cf32(0,0),  cf32(0,0),  cf32(0,0),  // bins 32..35
    cf32(1,0),  cf32(1,0),  cf32(-1,0), cf32(1,0),  // bins 36..39: sc -28..-25
    cf32(-1,0), cf32(1,0),  cf32(-1,0), cf32(-1,0), // bins 40..43
    cf32(-1,0), cf32(-1,0), cf32(-1,0), cf32(1,0),  // bins 44..47
    cf32(1,0),  cf32(-1,0), cf32(-1,0), cf32(1,0),  // bins 48..51
    cf32(-1,0), cf32(1,0),  cf32(-1,0), cf32(1,0),  // bins 52..55
    cf32(1,0),  cf32(1,0),  cf32(1,0),  cf32(-1,0), // bins 56..59: sc -8..-5
    cf32(-1,0), cf32(1,0),  cf32(-1,0), cf32(1,0),  // bins 60..63: sc -4..-1
};

// 127-period pilot polarity sequence from IEEE 802.11
const float PILOT_POLARITY[127] = {
     1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1,
    -1,-1, 1, 1,-1, 1, 1,-1, 1, 1, 1, 1, 1, 1,-1, 1,
     1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1,
    -1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,
    -1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1,
    -1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,
    -1,-1,-1,-1,-1, 1,-1, 1, 1,-1, 1,-1, 1, 1, 1,-1,
    -1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1
};

} // namespace ofdm
