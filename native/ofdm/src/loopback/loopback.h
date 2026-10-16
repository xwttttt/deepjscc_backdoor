#pragma once
#include "dsp/ofdm_core.h"
#include "tx/tx_pipeline.h"
#include "rx/rx_pipeline.h"
#include <vector>

namespace ofdm {

// Software loopback: TX output -> RX input without SDR hardware
// Adds optional channel impairments: AWGN, CFO, multipath
struct LoopbackConfig {
    float awgn_snr_db = 30.0f;      // AWGN SNR in dB
    float cfo_hz = 0.0f;            // Carrier frequency offset in Hz
    bool  multipath = false;         // Enable simple multipath
    int   random_delay = 0;          // Random sample delay [0..random_delay)
};

struct LoopbackResult {
    bool success;
    float max_error;           // max |recovered - original|
    float nmse_db;             // normalized MSE in dB
    float csi_nmse_db;         // CSI NMSE in dB
    RxMetadata rx_meta;
    std::vector<cf32> recovered;  // recovered data
};

// Run loopback test: transmit data, pass through channel, receive
LoopbackResult run_loopback(const cf32* data, size_t len, uint32_t burst_id,
                            const LoopbackConfig& cfg = {});

} // namespace ofdm
