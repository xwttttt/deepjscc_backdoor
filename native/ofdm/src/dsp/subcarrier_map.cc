#include "dsp/subcarrier_map.h"
#include <cmath>

namespace ofdm {

void map_data_to_freq(const cf32* data, cf32* freq) {
    // Zero all bins
    for (int i = 0; i < FFT_SIZE; i++) {
        freq[i] = cf32(0, 0);
    }
    // Map 52 data values to their FFT bin positions
    for (int i = 0; i < 52; i++) {
        int bin = sc_to_fft(DATA_SC_ORDER[i]);
        freq[bin] = data[i];
    }
}

void map_freq_to_data(const cf32* freq, cf32* data) {
    for (int i = 0; i < 52; i++) {
        int bin = sc_to_fft(DATA_SC_ORDER[i]);
        data[i] = freq[bin];
    }
}

void insert_pilots(cf32* freq, int pilot_polarity_idx) {
    float p = PILOT_POLARITY[pilot_polarity_idx % 127];
    // Wi-Fi signed-carrier order [-21,-7,+7,+21] is [+,+,+,-] * p.
    freq[7]  = cf32(+p, 0);
    freq[21] = cf32(-p, 0);
    freq[43] = cf32(+p, 0);
    freq[57] = cf32(+p, 0);
}

float extract_pilots_phase(const cf32* freq, int pilot_polarity_idx) {
    float p = PILOT_POLARITY[pilot_polarity_idx % 127];
    // Expected pilots in FFT-bin order 7,21,43,57: [+p,-p,+p,+p].
    // Phase error = angle of sum(received * conj(expected))
    cf32 sum = cf32(0, 0);
    sum += freq[7]  * cf32(+p, 0);
    sum += freq[21] * cf32(-p, 0);
    sum += freq[43] * cf32(+p, 0);
    sum += freq[57] * cf32(+p, 0);
    return std::arg(sum);
}

PilotTrack extract_pilots_full(const cf32* freq, int pilot_polarity_idx) {
    float p = PILOT_POLARITY[pilot_polarity_idx % 127];

    // Pilot subcarrier signed positions: -21, -7, +7, +21
    // FFT bins: 43, 57, 7, 21
    // Expected values in signed order: [+, +, +, -] * p
    int bins[4] = {43, 57, 7, 21};
    float expected[4] = {+p, +p, +p, -p};
    float sc_pos[4] = {-21.0f, -7.0f, +7.0f, +21.0f};

    // Compute phase at each pilot position (already sorted by sc_pos)
    float phases[4];
    for (int i = 0; i < 4; i++) {
        cf32 r = freq[bins[i]] * cf32(expected[i], 0);
        phases[i] = std::arg(r);
    }

    // Unwrap phases along subcarrier position order (already monotonic)
    for (int i = 1; i < 4; i++) {
        float diff = phases[i] - phases[i - 1];
        while (diff > M_PI)  diff -= 2.0f * M_PI;
        while (diff < -M_PI) diff += 2.0f * M_PI;
        phases[i] = phases[i - 1] + diff;
    }

    // Least-squares linear fit: phase(k) = a + b * sc_pos[k]
    float sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (int i = 0; i < 4; i++) {
        sum_x  += sc_pos[i];
        sum_y  += phases[i];
        sum_xx += sc_pos[i] * sc_pos[i];
        sum_xy += sc_pos[i] * phases[i];
    }

    float n = 4.0f;
    float denom = n * sum_xx - sum_x * sum_x;
    PilotTrack result;
    if (std::abs(denom) > 1e-10f) {
        result.phase_slope = (n * sum_xy - sum_x * sum_y) / denom;
        result.common_phase = (sum_y - result.phase_slope * sum_x) / n;
    } else {
        result.common_phase = sum_y / n;
        result.phase_slope = 0.0f;
    }
    return result;
}

} // namespace ofdm
