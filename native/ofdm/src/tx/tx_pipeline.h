#pragma once
#include "dsp/ofdm_core.h"
#include "dsp/control_header.h"
#include <vector>

namespace ofdm {

// Assemble a hybrid burst: a standard 802.11 legacy anchor carrying extension
// control metadata, then a project HT-LTF and raw complex OFDM blocks. Each
// block maps 52 consecutive input values onto the HT20 data subcarriers and
// inserts the normal four pilots; DC and guard carriers remain zero.
class TxPipeline {
public:
    struct Result {
        std::vector<cf32> samples;
        ControlHeader header;
        float waveform_scale{1.0f};
    };

    // Build a burst from input CF32 data
    Result build_burst(const cf32* data, size_t len, uint32_t burst_id);

    // Total number of 52-value OFDM blocks needed for len CF32 values
    static int calc_num_data_symbols(size_t len);

    // Total burst length in samples
    static int calc_burst_samples(int num_data_symbols);

private:
    void encode_data_symbols(const cf32* data, size_t len, int num_syms,
                             cf32* out, int pilot_start);
};

} // namespace ofdm
