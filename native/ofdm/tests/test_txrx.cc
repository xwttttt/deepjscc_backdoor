#include "test_framework.h"
#include "tx/tx_pipeline.h"
#include "rx/rx_pipeline.h"
#include "loopback/loopback.h"
#include "wifi/wifi_anchor.h"
#include "dsp/subcarrier_map.h"
#include <liquid/liquid.h>
#include <cmath>
#include <random>
#include <algorithm>

using namespace ofdm;

bool test_wifi_anchor() {
    const wifi_anchor::ExtensionControl control{0x12345678u, 100u, 2u};
    const auto metadata = wifi_anchor::encode_extension_control(control);
    const auto waveform = wifi_anchor::build(metadata);
    const auto decoded = wifi_anchor::decode(waveform.data(), waveform.size());
    TEST_ASSERT(decoded.ok, "legacy Wi-Fi anchor should decode");
    TEST_ASSERT(decoded.frame_samples == wifi_anchor::kFrameSamples, "anchor length");
    TEST_ASSERT(decoded.metadata == metadata, "anchor metadata/FCS");
    wifi_anchor::ExtensionControl recovered;
    TEST_ASSERT(wifi_anchor::decode_extension_control(decoded.metadata, recovered), "extension control");
    TEST_ASSERT(recovered.burst_id == control.burst_id, "32-bit anchor burst id");
    TEST_ASSERT(recovered.original_length == control.original_length, "anchor original length");
    TEST_ASSERT(recovered.num_data_symbols == control.num_data_symbols, "anchor OFDM block count");
    return true;
}

bool test_frequency_block_mapping() {
    std::vector<cf32> data(53);
    for (std::size_t index = 0; index < data.size(); ++index) {
        data[index] = cf32(0.01f * static_cast<float>(index + 1),
                           -0.005f * static_cast<float>(index + 1));
    }
    const auto burst = TxPipeline{}.build_burst(data.data(), data.size(), 0x89abcdefu);
    TEST_ASSERT(burst.header.num_data_symbols == 2, "53 values require two OFDM blocks");

    const std::size_t data_start = wifi_anchor::kFrameSamples + wifi_anchor::kGuardSamples + SYM_SAMP;
    for (int symbol = 0; symbol < 2; ++symbol) {
        cf32 frequency[FFT_SIZE]{};
        auto plan = fft_create_plan(
            FFT_SIZE,
            reinterpret_cast<liquid_float_complex*>(const_cast<cf32*>(
                burst.samples.data() + data_start + symbol * SYM_SAMP + CP_LEN)),
            reinterpret_cast<liquid_float_complex*>(frequency),
            LIQUID_FFT_FORWARD, 0);
        fft_execute(plan);
        fft_destroy_plan(plan);
        const float scale = static_cast<float>(FFT_SIZE) /
            7.211102550927978f * burst.waveform_scale;
        for (int position = 0; position < NUM_DATA_SC; ++position) {
            const std::size_t source = static_cast<std::size_t>(symbol * NUM_DATA_SC + position);
            const cf32 expected = source < data.size() ? data[source] * scale : cf32{};
            TEST_ASSERT(std::abs(frequency[data_pos_to_fft(position)] - expected) < 1e-4f,
                        "network values map to consecutive 52-bin OFDM blocks with zero padding");
        }
        TEST_ASSERT(std::abs(frequency[0]) < 1e-4f, "DC bin remains zero");
        for (const int bin : PILOT_POS) TEST_ASSERT(std::abs(frequency[bin]) > 0.1f, "pilot is inserted");
    }
    return true;
}

// Test TX burst assembly
bool test_tx_burst_structure() {
    // Generate test data
    std::vector<cf32> data(1024);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = cf32(std::cos(0.01f * i), std::sin(0.01f * i));
    }

    TxPipeline tx;
    auto result = tx.build_burst(data.data(), data.size(), 1);

    // Check burst structure
    int expected_syms = TxPipeline::calc_num_data_symbols(data.size());
    TEST_ASSERT(result.header.burst_id == 1, "burst_id");
    TEST_ASSERT(result.header.original_length == 1024, "original_length");
    TEST_ASSERT(result.header.num_data_symbols == expected_syms, "num_data_symbols");

    int expected_samples = TxPipeline::calc_burst_samples(expected_syms);
    TEST_ASSERT((int)result.samples.size() == expected_samples, "total samples");
    return true;
}

// Test TX/RX golden loopback
bool test_golden_loopback() {
    // Small data that fits in one symbol
    std::vector<cf32> data(32);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = cf32(0.5f * std::cos(0.1f * i), 0.5f * std::sin(0.1f * i));
    }

    LoopbackConfig cfg;
    cfg.awgn_snr_db = 100.0f;  // no noise
    cfg.cfo_hz = 0.0f;
    cfg.multipath = false;
    cfg.random_delay = 0;

    auto result = run_loopback(data.data(), data.size(), 42, cfg);
    TEST_ASSERT(result.success, "golden loopback should succeed");
    TEST_ASSERT(result.rx_meta.burst_id == 42, "burst_id");
    TEST_ASSERT(result.rx_meta.original_length == 32, "original_length");
    TEST_ASSERT(result.rx_meta.crc_ok, "CRC should pass");
    TEST_ASSERT(result.max_error < 1e-4f, "max error < 1e-4 for golden");
    TEST_ASSERT(result.csi_nmse_db < -60.0f, "ideal HT-LTF CSI should be flat");
    return true;
}

// Test loopback with AWGN
bool test_awgn_loopback() {
    std::vector<cf32> data(256);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = cf32(dist(rng), dist(rng));
    }

    LoopbackConfig cfg;
    cfg.awgn_snr_db = 30.0f;
    cfg.cfo_hz = 0.0f;
    cfg.multipath = false;

    auto result = run_loopback(data.data(), data.size(), 7, cfg);
    TEST_ASSERT(result.success, "AWGN loopback should succeed");
    TEST_ASSERT(result.rx_meta.crc_ok, "CRC should pass at 30dB SNR");
    TEST_ASSERT(result.nmse_db < -20.0f, "NMSE should be < -20dB");
    TEST_ASSERT(result.csi_nmse_db < -15.0f, "CSI NMSE should be < -15dB");
    return true;
}

// Test loopback with CFO
bool test_cfo_loopback() {
    std::vector<cf32> data(512);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = cf32(0.5f * std::cos(0.05f * i), 0.5f * std::sin(0.05f * i));
    }

    LoopbackConfig cfg;
    cfg.awgn_snr_db = 100.0f;
    cfg.cfo_hz = 5000.0f;  // 5 kHz CFO
    cfg.multipath = false;

    auto result = run_loopback(data.data(), data.size(), 99, cfg);
    TEST_ASSERT(result.success, "CFO loopback should succeed");
    TEST_ASSERT(result.rx_meta.crc_ok, "CRC should pass with 5kHz CFO");
    return true;
}

// Regression test for the Marconi-style STF plateau + L-LTF timing
// refinement.  This approximates an SDR capture that starts with unrelated
// samples, has a short multipath channel, and has residual frequency offset.
bool test_rf_like_sync() {
    std::vector<cf32> data(300);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = cf32(0.35f * std::cos(0.037f * i),
                       0.25f * std::sin(0.071f * i));
    }

    LoopbackConfig cfg;
    cfg.awgn_snr_db = 25.0f;
    cfg.cfo_hz = 3500.0f;
    cfg.multipath = true;
    cfg.random_delay = 127;

    const auto result = run_loopback(data.data(), data.size(), 1234, cfg);
    TEST_ASSERT(result.success, "RF-like timing loopback should decode");
    TEST_ASSERT(result.rx_meta.crc_ok, "RF-like control CRC should pass");
    TEST_ASSERT(result.recovered.size() == data.size(), "RF-like recovered length");
    return true;
}

// Feed the receiver in small streaming callbacks, as the SDR backend does,
// rather than giving it one complete capture buffer.
bool test_streaming_rx() {
    std::vector<cf32> data(256);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = cf32(0.25f * std::cos(0.021f * i), 0.25f * std::sin(0.043f * i));
    }
    TxPipeline tx;
    const auto burst = tx.build_burst(data.data(), data.size(), 77);
    std::vector<cf32> capture(47, cf32(0.0f, 0.0f));
    capture.insert(capture.end(), burst.samples.begin(), burst.samples.end());

    RxPipeline rx;
    std::vector<RxResult> results;
    for (std::size_t offset = 0; offset < capture.size();) {
        const auto count = std::min<std::size_t>(257, capture.size() - offset);
        auto current = rx.process(capture.data() + offset, static_cast<int>(count));
        results.insert(results.end(), current.begin(), current.end());
        offset += count;
    }
    TEST_ASSERT(results.size() == 1, "streaming receiver should decode one burst");
    TEST_ASSERT(results[0].meta.burst_id == 77, "streaming burst id");
    TEST_ASSERT(results[0].data.size() == data.size(), "streaming recovered length");
    return true;
}

// Test multi-burst
bool test_multi_burst() {
    // Two bursts back-to-back
    std::vector<cf32> data1(64), data2(128);
    for (size_t i = 0; i < data1.size(); i++) data1[i] = cf32(0.1f * i, 0);
    for (size_t i = 0; i < data2.size(); i++) data2[i] = cf32(0, 0.1f * i);

    TxPipeline tx;
    auto burst1 = tx.build_burst(data1.data(), data1.size(), 100);
    auto burst2 = tx.build_burst(data2.data(), data2.size(), 101);

    // Concatenate
    std::vector<cf32> combined;
    combined.insert(combined.end(), burst1.samples.begin(), burst1.samples.end());
    combined.insert(combined.end(), burst2.samples.begin(), burst2.samples.end());

    RxPipeline rx;
    auto results = rx.process(combined.data(), (int)combined.size());
    TEST_ASSERT(results.size() >= 2, "should decode both bursts");
    if (results.size() >= 2) {
        TEST_ASSERT(results[0].meta.burst_id == 100, "first burst id");
        TEST_ASSERT(results[1].meta.burst_id == 101, "second burst id");
    }
    return true;
}

// Test padding/trimming
bool test_padding_trim() {
    // Data that doesn't align to 52
    std::vector<cf32> data(100);
    for (size_t i = 0; i < data.size(); i++) data[i] = cf32(1.0f, -1.0f);

    LoopbackConfig cfg;
    cfg.awgn_snr_db = 100.0f;

    auto result = run_loopback(data.data(), data.size(), 10, cfg);
    TEST_ASSERT(result.success, "padding loopback should succeed");
    TEST_ASSERT(result.recovered.size() == data.size(), "recovered length matches original");
    return true;
}

void run_txrx_tests() {
    RUN_TEST(test_wifi_anchor);
    RUN_TEST(test_frequency_block_mapping);
    RUN_TEST(test_tx_burst_structure);
    RUN_TEST(test_golden_loopback);
    RUN_TEST(test_awgn_loopback);
    RUN_TEST(test_cfo_loopback);
    RUN_TEST(test_rf_like_sync);
    RUN_TEST(test_streaming_rx);
    RUN_TEST(test_multi_burst);
    RUN_TEST(test_padding_trim);
}
