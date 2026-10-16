#include "test_framework.h"
#include "dsp/ofdm_core.h"
#include "dsp/subcarrier_map.h"
#include "dsp/crc16.h"
#include "dsp/control_header.h"
#include "dsp/preamble.h"
#include <liquid/liquid.h>
#include <cmath>
#include <cstring>
#include <numeric>
#include <algorithm>

using namespace ofdm;

// Test 52->64 mapping and inverse
bool test_subcarrier_map_roundtrip() {
    cf32 data_in[52];
    for (int i = 0; i < 52; i++) {
        data_in[i] = cf32((float)(i + 1), (float)(i + 1) * 0.5f);
    }

    cf32 freq[64];
    map_data_to_freq(data_in, freq);

    // Verify data is at correct bins
    for (int i = 0; i < 52; i++) {
        int bin = sc_to_fft(DATA_SC_ORDER[i]);
        TEST_ASSERT(std::abs(freq[bin].real() - data_in[i].real()) < 1e-6f,
                     "data bin mismatch at pos " + std::to_string(i));
    }

    // Verify DC and guard bins are zero
    TEST_ASSERT(std::abs(freq[0]) < 1e-6f, "DC should be zero");
    TEST_ASSERT(std::abs(freq[32]) < 1e-6f, "guard bin 32 should be zero");
    for (int b : {29, 30, 31, 33, 34, 35}) {
        TEST_ASSERT(std::abs(freq[b]) < 1e-6f, "guard bin " + std::to_string(b) + " should be zero");
    }

    // Inverse mapping
    cf32 data_out[52];
    map_freq_to_data(freq, data_out);

    for (int i = 0; i < 52; i++) {
        TEST_ASSERT_NEAR(data_out[i].real(), data_in[i].real(), 1e-6f, "roundtrip real");
        TEST_ASSERT_NEAR(data_out[i].imag(), data_in[i].imag(), 1e-6f, "roundtrip imag");
    }
    return true;
}

// Test pilot insertion and polarity
bool test_pilot_polarity() {
    cf32 freq[64] = {};
    for (int p = 0; p < 10; p++) {
        std::fill(std::begin(freq), std::end(freq), cf32{});
        insert_pilots(freq, p);

        float expected_p = PILOT_POLARITY[p % 127];
        TEST_ASSERT_NEAR(std::abs(freq[7]), std::abs(expected_p), 1e-6f, "pilot 7 magnitude");
        TEST_ASSERT_NEAR(std::abs(freq[21]), std::abs(expected_p), 1e-6f, "pilot 21 magnitude");
        TEST_ASSERT_NEAR(std::abs(freq[43]), std::abs(expected_p), 1e-6f, "pilot 43 magnitude");
        TEST_ASSERT_NEAR(std::abs(freq[57]), std::abs(expected_p), 1e-6f, "pilot 57 magnitude");

        // Check signed-carrier order [-21,-7,+7,+21] = [+, +, +, -].
        float sign7 = freq[7].real() > 0 ? 1.0f : -1.0f;
        float sign21 = freq[21].real() > 0 ? 1.0f : -1.0f;
        float sign43 = freq[43].real() > 0 ? 1.0f : -1.0f;
        float sign57 = freq[57].real() > 0 ? 1.0f : -1.0f;
        TEST_ASSERT_NEAR(sign7 * sign21, -1.0f, 0.1f, "pilots +7,+21 opposite sign");
        TEST_ASSERT_NEAR(sign7 * sign43, 1.0f, 0.1f, "pilots 7,43 same sign");
        TEST_ASSERT_NEAR(sign7 * sign57, 1.0f, 0.1f, "pilots +7,-7 same sign");
    }
    return true;
}

// Test CRC-16
bool test_crc16() {
    // Known test vector
    uint8_t bits[104];
    for (int i = 0; i < 104; i++) bits[i] = (i * 7 + 3) & 1;

    uint16_t crc = compute_crc16(bits, 104);
    TEST_ASSERT(crc != 0, "CRC should not be zero for random data");

    // Build full 120-bit vector with CRC appended
    uint8_t full[120];
    memcpy(full, bits, 104);
    for (int i = 0; i < 16; i++) {
        full[104 + i] = (crc >> (15 - i)) & 1;
    }
    TEST_ASSERT(verify_crc16(full, 120), "CRC verification should pass");

    // Corrupt one bit
    full[50] ^= 1;
    TEST_ASSERT(!verify_crc16(full, 120), "CRC should fail after corruption");
    return true;
}

// Test control header encode/decode
bool test_control_header() {
    ControlHeader hdr_in;
    hdr_in.magic = CTRL_HDR_MAGIC;
    hdr_in.version_flags = 0x42;
    hdr_in.burst_id = 12345;
    hdr_in.original_length = 1024;
    hdr_in.num_data_symbols = 20;

    uint8_t bits[CTRL_HDR_TOTAL_BITS];
    encode_control_header(hdr_in, bits);

    ControlHeader hdr_out;
    bool ok = decode_control_header(bits, hdr_out);
    TEST_ASSERT(ok, "control header decode should succeed");
    TEST_ASSERT(hdr_out.magic == CTRL_HDR_MAGIC, "magic mismatch");
    TEST_ASSERT(hdr_out.version_flags == 0x42, "version mismatch");
    TEST_ASSERT(hdr_out.burst_id == 12345, "burst_id mismatch");
    TEST_ASSERT(hdr_out.original_length == 1024, "length mismatch");
    TEST_ASSERT(hdr_out.num_data_symbols == 20, "num_symbols mismatch");
    return true;
}

// Test FFT/IFFT scaling
bool test_fft_ifft_scale() {
    cf32 data[64];
    for (int i = 0; i < 64; i++) {
        data[i] = cf32(std::cos(0.1f * i), std::sin(0.1f * i));
    }

    // IFFT then FFT should recover original
    cf32 freq[64], time[64];

    fftplan fwd = fft_create_plan(64,
        reinterpret_cast<liquid_float_complex*>(data),
        reinterpret_cast<liquid_float_complex*>(freq),
        LIQUID_FFT_FORWARD, 0);
    fft_execute(fwd);
    fft_destroy_plan(fwd);

    fftplan bwd = fft_create_plan(64,
        reinterpret_cast<liquid_float_complex*>(freq),
        reinterpret_cast<liquid_float_complex*>(time),
        LIQUID_FFT_BACKWARD, 0);
    fft_execute(bwd);
    fft_destroy_plan(bwd);

    // liquid's FFT is unnormalized, so IFFT(FFT(x)) = N*x
    for (int i = 0; i < 64; i++) {
        time[i] /= 64.0f;
        TEST_ASSERT_NEAR(time[i].real(), data[i].real(), 1e-4f, "FFT/IFFT roundtrip real");
        TEST_ASSERT_NEAR(time[i].imag(), data[i].imag(), 1e-4f, "FFT/IFFT roundtrip imag");
    }
    return true;
}

// Test preamble generation
bool test_preamble() {
    cf32 stf[160], ltf[160], ht_ltf[80];
    generate_stf(stf);
    generate_ltf(ltf);
    generate_ht_ltf(ht_ltf);

    // STF should be periodic with period 16
    for (int i = 0; i < 16; i++) {
        float err = std::abs(stf[i] - stf[i + 16]);
        TEST_ASSERT(err < 1e-4f, "STF periodicity");
    }

    // LTF symbols should be identical (after CP)
    for (int i = 0; i < 64; i++) {
        float err = std::abs(ltf[32 + i] - ltf[96 + i]);
        TEST_ASSERT(err < 1e-4f, "LTF symbol identity");
    }

    // Check non-zero power
    float stf_power = 0, ltf_power = 0;
    for (int i = 0; i < 64; i++) {
        stf_power += std::norm(stf[i]);
        ltf_power += std::norm(ltf[32 + i]);
    }
    TEST_ASSERT(stf_power > 0.01f, "STF has power");
    TEST_ASSERT(ltf_power > 0.01f, "LTF has power");

    int ht_occupied = 0;
    for (int bin = 0; bin < FFT_SIZE; ++bin) {
        if (std::abs(HT_LTF_F[bin]) > 0.1f) ++ht_occupied;
    }
    TEST_ASSERT(ht_occupied == NUM_USED_SC, "HT-LTF has exactly 56 occupied bins");
    TEST_ASSERT(std::abs(HT_LTF_F[0]) < 1e-6f, "HT-LTF DC is zero");
    for (const int bin : {29, 30, 31, 32, 33, 34, 35}) {
        TEST_ASSERT(std::abs(HT_LTF_F[bin]) < 1e-6f, "HT-LTF guard bin is zero");
    }
    return true;
}

// Test CSI estimation
bool test_csi_estimate() {
    // Ideal loopback: CSI should be all ones on data/pilot subcarriers
    cf32 ltf[160];
    generate_ltf(ltf);

    // FFT the first LTF symbol (skip CP=32)
    cf32 freq[64];
    fftplan q = fft_create_plan(64,
        reinterpret_cast<liquid_float_complex*>(&ltf[32]),
        reinterpret_cast<liquid_float_complex*>(freq),
        LIQUID_FFT_FORWARD, 0);
    fft_execute(q);
    fft_destroy_plan(q);

    // CSI = received / known L_LTF
    for (int i = 0; i < 64; i++) {
        if (std::abs(L_LTF_F[i]) > 0.1f) {
            float csi_mag = std::abs(freq[i] / L_LTF_F[i]);
            TEST_ASSERT_NEAR(csi_mag, 1.0f, 0.1f, "CSI magnitude near 1 in loopback");
        }
    }
    return true;
}

// Test ZF equalization
bool test_zf_equalize() {
    // Simple test: known channel H=2, transmit X, receive Y=2X, ZF should recover X
    cf32 freq[64];
    for (int i = 0; i < 64; i++) {
        freq[i] = cf32(2.0f, 0.0f) * cf32((float)(i % 5) - 2.0f, (float)(i % 3) - 1.0f);
    }
    cf32 csi[64];
    for (int i = 0; i < 64; i++) csi[i] = cf32(2.0f, 0.0f);

    // ZF: Y/H = X
    for (int i = 0; i < 64; i++) {
        cf32 eq = freq[i] / csi[i];
        cf32 expected((float)(i % 5) - 2.0f, (float)(i % 3) - 1.0f);
        TEST_ASSERT_NEAR(eq.real(), expected.real(), 1e-4f, "ZF real");
        TEST_ASSERT_NEAR(eq.imag(), expected.imag(), 1e-4f, "ZF imag");
    }
    return true;
}

// Test pilot phase extraction
bool test_phase_slope_compensation() {
    // Create known equalized pilots with a phase slope
    cf32 freq[64] = {};
    float slope = 0.01f;  // radians per subcarrier
    float p = PILOT_POLARITY[0];

    // Must match the bin-to-sc_pos mapping used by extract_pilots_full:
    // bins[0]=43(sc=-21), bins[1]=57(sc=-7), bins[2]=7(sc=+7), bins[3]=21(sc=+21)
    int bins[4] = {43, 57, 7, 21};
    float sc_pos[4] = {-21.0f, -7.0f, +7.0f, +21.0f};
    float expected_val[4] = {+p, +p, +p, -p};

    for (int i = 0; i < 4; i++) {
        float phase = slope * sc_pos[i];
        freq[bins[i]] = cf32(expected_val[i] * std::cos(phase), expected_val[i] * std::sin(phase));
    }

    auto track = extract_pilots_full(freq, 0);
    TEST_ASSERT(std::abs(track.phase_slope - slope) < 0.005f, "phase slope estimation");
    TEST_ASSERT(std::abs(track.common_phase) < 0.01f, "common phase near zero");
    return true;
}

void run_dsp_tests() {
    RUN_TEST(test_subcarrier_map_roundtrip);
    RUN_TEST(test_pilot_polarity);
    RUN_TEST(test_crc16);
    RUN_TEST(test_control_header);
    RUN_TEST(test_fft_ifft_scale);
    RUN_TEST(test_preamble);
    RUN_TEST(test_csi_estimate);
    RUN_TEST(test_zf_equalize);
    RUN_TEST(test_phase_slope_compensation);
}
