#pragma once
#include "dsp/ofdm_core.h"

namespace ofdm {

// Control header: 120 bits total
// magic(16) | version_flags(8) | burst_id(32) | original_length(32) | num_data_symbols(16) | crc16(16)
constexpr int CTRL_HDR_MAGIC         = 0x4F44; // "OD" for OFDM
constexpr int CTRL_HDR_TOTAL_BITS    = 120;
constexpr int CTRL_HDR_DATA_BITS     = 104;  // without CRC
constexpr int CTRL_HDR_NUM_SYMBOLS   = 3;    // ceil(120 / 52)

struct ControlHeader {
    uint16_t magic;
    uint8_t  version_flags;
    uint32_t burst_id;
    uint32_t original_length;
    uint16_t num_data_symbols;
    uint16_t crc;
};

// Encode control header to 120 bits
void encode_control_header(const ControlHeader& hdr, uint8_t* bits_out);

// Decode control header from 120 bits; returns false if magic or CRC mismatch
bool decode_control_header(const uint8_t* bits_in, ControlHeader& hdr);

// Encode control header bits to 3 BPSK OFDM symbols (time domain, with CP)
// out: 3 * 80 = 240 samples
// pilot_start: pilot polarity starting index (should be 0 for first data symbol after LTF)
void encode_ctrl_symbols(const uint8_t* bits, cf32* out, int pilot_start);

// Decode 3 BPSK OFDM symbols to bits (after FFT and channel equalization)
// in: freq-domain samples for 3 symbols, each 64 bins
// channel_est: CSI from LTF
void decode_ctrl_symbols(const cf32* sym_freq, const cf32* channel_est,
                          uint8_t* bits_out, int pilot_start);

} // namespace ofdm
