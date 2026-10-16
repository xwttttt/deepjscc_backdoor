#pragma once
#include "dsp/ofdm_core.h"
#include "dsp/control_header.h"
#include <vector>

namespace ofdm {

struct RxMetadata {
    uint32_t burst_id;
    uint32_t original_length;
    int      num_data_symbols;
    float    cfo_hz;          // carrier frequency offset in Hz
    float    rssi;            // received signal strength
    float    snr_db;          // signal-to-noise ratio in dB
    float    csi_quality_db;  // HT-LTF channel-estimate quality gate
    float    noise_var;       // estimated noise variance
    bool     crc_ok;          // Wi-Fi anchor FCS (or legacy control CRC)
    int      sync_offset;     // sample index of sync point
};

struct RxResult {
    bool                    success;
    RxMetadata              meta;
    cf32                    csi[FFT_SIZE];   // channel state estimate
    std::vector<cf32>       data;            // recovered CF32 data
};

struct RxDiagnostics {
    uint64_t stf_candidates{};
    uint64_t ltf_locks{};
    uint64_t csi_rejects{};
    uint64_t control_failures{};
    uint64_t header_failures{};
    uint64_t decoded{};
    float last_csi_quality_db{-100.0f};
    float max_csi_quality_db{-100.0f};
    float last_cfo_hz{};
};

class RxPipeline {
public:
    RxPipeline();

    // Process a raw sample buffer and return recovered bursts
    // samples: input CF32 samples at 20 Msps
    // num_samples: number of input samples
    // Returns all successfully decoded bursts
    std::vector<RxResult> process(const cf32* samples, int num_samples);

    // Reset streaming synchronizer and buffered samples.
    void reset();

    void set_mmse(bool enabled) { use_mmse_ = enabled; }
    RxDiagnostics diagnostics() const { return diagnostics_; }

private:
    // Detection state
    enum State { ST_DETECT, ST_SYNC, ST_DEMOD };

    // STF autocorrelation detection (from trigger_impl.cc)
    // Returns index of detected burst start, or -1 if not found
    int detect_stf(const cf32* samples, int num_samples, float& ac_peak);

    // Coarse CFO from STF autocorrelation (from sync_impl.cc)
    float estimate_cfo_coarse(const cf32* samples, int stf_end_idx);

    // Fine CFO from LTF cross-correlation (from sync_impl.cc)
    float estimate_cfo_fine(const cf32* samples, int ltf_start_idx, float coarse_cfo);

    // Channel estimation from L-LTF/HT-LTF. New extension bursts prefer the
    // immediately preceding HT-LTF for all 56 occupied subcarriers.
    void estimate_channel(const cf32* samples, int ltf_start_idx, int hltf_start_idx,
                          float cfo_rad, cf32* csi, bool prefer_ht = false);

    // Noise variance estimation from repeated LTF symbols
    float estimate_noise_var(const cf32* samples, int ltf_start_idx, float cfo_rad,
                             const cf32* csi);

    // MMSE equalization of one OFDM symbol
    // time_in: 80 samples (CP + FFT) of time domain
    // cfo_rad: CFO in radians/sample
    // csi: channel estimate [64]
    // pilot_idx: pilot polarity index
    // noise_var: noise variance
    // eq_out: equalized frequency domain [64]
    void equalize_symbol(const cf32* time_in, float cfo_rad, const cf32* csi,
                          int pilot_idx, float noise_var, cf32* eq_out);

    // ZF equalization (diagnostic mode)
    void equalize_symbol_zf(const cf32* time_in, float cfo_rad, const cf32* csi,
                             int pilot_idx, cf32* eq_out);

    // CFO compensation: multiply by exp(-j*n*cfo_rad)
    void compensate_cfo(cf32* samples, int num, int sample_offset, float cfo_rad);

    // FFT helper
    void fft_64(const cf32* in, cf32* out);
    void ifft_64(const cf32* in, cf32* out);

    bool use_mmse_ = true;
    std::vector<cf32> buffer_;
    std::size_t scan_offset_ = 0;
    cf32 dc_{};
    RxDiagnostics diagnostics_{};
};

} // namespace ofdm
