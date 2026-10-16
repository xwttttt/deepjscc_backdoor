#pragma once
#include "dsp/ofdm_core.h"

namespace ofdm {

// Compute CRC-16/CCITT over a bit array
// Returns 16-bit CRC value
uint16_t compute_crc16(const uint8_t* bits, int num_bits);

// Verify CRC-16: last 16 bits of data should match CRC of preceding bits
bool verify_crc16(const uint8_t* bits, int total_bits);

} // namespace ofdm
