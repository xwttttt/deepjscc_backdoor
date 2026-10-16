#include "dsp/crc16.h"

namespace ofdm {

// CRC-16/CCITT: polynomial 0x1021, init 0xFFFF
uint16_t compute_crc16(const uint8_t* bits, int num_bits) {
    uint16_t crc = 0xFFFF;
    for (int i = 0; i < num_bits; i++) {
        uint8_t bit = bits[i] & 1;
        uint8_t msb = (crc >> 15) & 1;
        crc <<= 1;
        if (msb ^ bit) {
            crc ^= 0x1021;
        }
    }
    return crc;
}

bool verify_crc16(const uint8_t* bits, int total_bits) {
    if (total_bits < 16) return false;
    int data_bits = total_bits - 16;
    uint16_t computed = compute_crc16(bits, data_bits);
    // Extract stored CRC from last 16 bits (MSB first)
    uint16_t stored = 0;
    for (int i = 0; i < 16; i++) {
        stored = (stored << 1) | (bits[data_bits + i] & 1);
    }
    return computed == stored;
}

} // namespace ofdm
