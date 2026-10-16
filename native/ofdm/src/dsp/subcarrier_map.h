#pragma once
#include "dsp/ofdm_core.h"

namespace ofdm {

// Map 52 data values into 64-point FFT bins (data subcarriers only)
// Input: data[52] in signed-subcarrier order (-28 to +28, skipping DC, pilots, guards)
// Output: freq[64] in FFT bin order; pilots, DC, guards are zero
void map_data_to_freq(const cf32* data, cf32* freq);

// Map 64-point FFT output back to 52 data values
// Input: freq[64] in FFT bin order
// Output: data[52] in signed-subcarrier order
void map_freq_to_data(const cf32* freq, cf32* data);

// Insert pilot subcarriers into freq[64] at positions ±7, ±21
// pilot_polarity_idx: index into PILOT_POLARITY (incremented per symbol)
void insert_pilots(cf32* freq, int pilot_polarity_idx);

// Extract and track pilots from freq[64] after equalization
// Returns common phase error estimate from pilot tracking
float extract_pilots_phase(const cf32* freq, int pilot_polarity_idx);

// Extract pilots with both common phase and linear phase slope estimation
// Returns {common_phase, phase_slope} as a pair
struct PilotTrack {
    float common_phase;  // radians
    float phase_slope;   // radians per subcarrier
};
PilotTrack extract_pilots_full(const cf32* freq, int pilot_polarity_idx);

} // namespace ofdm
