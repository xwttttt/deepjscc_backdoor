#include "dsp/control_header.h"
#include "dsp/crc16.h"
#include "dsp/subcarrier_map.h"
#include <liquid/liquid.h>
#include <cstring>

namespace ofdm {

void encode_control_header(const ControlHeader& hdr, uint8_t* bits_out) {
    // Pack fields into bit array, MSB first
    int pos = 0;
    auto put_bits = [&](uint64_t val, int nbits) {
        for (int i = nbits - 1; i >= 0; i--) {
            bits_out[pos++] = (val >> i) & 1;
        }
    };
    put_bits(hdr.magic, 16);
    put_bits(hdr.version_flags, 8);
    put_bits(hdr.burst_id, 32);
    put_bits(hdr.original_length, 32);
    put_bits(hdr.num_data_symbols, 16);
    // Compute CRC over first 104 bits
    uint16_t crc = compute_crc16(bits_out, CTRL_HDR_DATA_BITS);
    put_bits(crc, 16);
}

bool decode_control_header(const uint8_t* bits_in, ControlHeader& hdr) {
    int pos = 0;
    auto get_bits = [&](int nbits) -> uint64_t {
        uint64_t val = 0;
        for (int i = 0; i < nbits; i++) {
            val = (val << 1) | (bits_in[pos++] & 1);
        }
        return val;
    };
    hdr.magic            = (uint16_t)get_bits(16);
    hdr.version_flags    = (uint8_t)get_bits(8);
    hdr.burst_id         = (uint32_t)get_bits(32);
    hdr.original_length  = (uint32_t)get_bits(32);
    hdr.num_data_symbols = (uint16_t)get_bits(16);
    hdr.crc              = (uint16_t)get_bits(16);

    if (hdr.magic != CTRL_HDR_MAGIC) return false;
    return verify_crc16(bits_in, CTRL_HDR_TOTAL_BITS);
}

// IFFT helper (unnormalized, matching liquid convention)
static void ifft_64_inplace(cf32* freq, cf32* time) {
    fftplan q = fft_create_plan(64,
        reinterpret_cast<liquid_float_complex*>(freq),
        reinterpret_cast<liquid_float_complex*>(time),
        LIQUID_FFT_BACKWARD, 0);
    fft_execute(q);
    fft_destroy_plan(q);
    for (int i = 0; i < FFT_SIZE; ++i) time[i] /= static_cast<float>(FFT_SIZE);
}

void encode_ctrl_symbols(const uint8_t* bits, cf32* out, int pilot_start) {
    for (int sym = 0; sym < CTRL_HDR_NUM_SYMBOLS; sym++) {
        cf32 freq[64] = {};
        int bit_offset = sym * NUM_DATA_SC;
        int bits_remaining = CTRL_HDR_TOTAL_BITS - bit_offset;
        int bits_this_sym = bits_remaining < NUM_DATA_SC ? bits_remaining : NUM_DATA_SC;

        // BPSK: bit 0 -> +1, bit 1 -> -1
        for (int i = 0; i < bits_this_sym; i++) {
            int bin = sc_to_fft(DATA_SC_ORDER[i]);
            freq[bin] = bits[bit_offset + i] ? cf32(-1, 0) : cf32(1, 0);
        }

        // Insert pilots
        insert_pilots(freq, pilot_start + sym);

        // IFFT
        cf32 time[64];
        ifft_64_inplace(freq, time);

        // Add CP (last 16 samples)
        cf32* sym_out = out + sym * SYM_SAMP;
        for (int i = 0; i < CP_LEN; i++) {
            sym_out[i] = time[FFT_SIZE - CP_LEN + i];
        }
        memcpy(&sym_out[CP_LEN], time, FFT_SIZE * sizeof(cf32));
    }
}

void decode_ctrl_symbols(const cf32* sym_freq, const cf32* channel_est,
                          uint8_t* bits_out, int pilot_start) {
    for (int sym = 0; sym < CTRL_HDR_NUM_SYMBOLS; sym++) {
        const cf32* freq = sym_freq + sym * FFT_SIZE;

        // ZF equalization
        cf32 eq[64];
        for (int i = 0; i < FFT_SIZE; i++) {
            if (std::abs(channel_est[i]) > 1e-6f) {
                eq[i] = freq[i] / channel_est[i];
            } else {
                eq[i] = cf32(0, 0);
            }
        }

        // Marconi tracks both common phase error and the linear phase slope
        // caused by residual timing/frequency offset.  Correcting only one
        // common phase leaves the outer BPSK subcarriers rotated and makes
        // the 120-bit control header fail its CRC even when the packet is
        // otherwise synchronized.
        const auto track = extract_pilots_full(eq, pilot_start + sym);
        for (int bin = 0; bin < FFT_SIZE; ++bin) {
            const float phase = -(track.common_phase + track.phase_slope * static_cast<float>(fft_to_sc(bin)));
            eq[bin] *= std::polar(1.0f, phase);
        }

        // Extract data subcarriers and hard-decision BPSK
        int bit_offset = sym * NUM_DATA_SC;
        int bits_remaining = CTRL_HDR_TOTAL_BITS - bit_offset;
        int bits_this_sym = bits_remaining < NUM_DATA_SC ? bits_remaining : NUM_DATA_SC;

        for (int i = 0; i < bits_this_sym; i++) {
            int bin = sc_to_fft(DATA_SC_ORDER[i]);
            bits_out[bit_offset + i] = eq[bin].real() < 0 ? 1 : 0;
        }
    }
}

} // namespace ofdm
