#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace ofdm::radio {

using Sample = std::complex<float>;

struct Range { double minimum{}; double maximum{}; double step{}; };

struct RadioChannelInfo {
    std::size_t index{};
    std::string label;
    std::vector<std::string> antennas;
    std::vector<Range> sample_rates;
    std::vector<Range> frequencies;
    std::vector<Range> gains;
    std::vector<Range> bandwidths;
    bool full_duplex{};
};

struct DeviceInfo {
    std::string id;
    std::string driver;
    std::string label;
    std::string serial;
    std::string device_args;
    std::vector<std::string> antennas;
    std::vector<std::string> rx_antennas;
    std::vector<std::string> tx_antennas;
    std::vector<RadioChannelInfo> rx_channel_info;
    std::vector<RadioChannelInfo> tx_channel_info;
    std::vector<Range> sample_rates;
    std::vector<Range> rx_sample_rates;
    std::vector<Range> tx_sample_rates;
    std::vector<Range> frequencies;
    std::vector<Range> rx_frequencies;
    std::vector<Range> tx_frequencies;
    std::vector<Range> gains;
    std::vector<Range> rx_gains;
    std::vector<Range> tx_gains;
    std::vector<Range> rx_bandwidths;
    std::vector<Range> tx_bandwidths;
    std::size_t rx_channels{};
    std::size_t tx_channels{};
    bool hardware_time{};
    bool rx{};
    bool tx{};
    bool full_duplex{};
};

struct DeviceDiagnostics {
    bool soapy_available{};
    bool modules_found{};
    bool drivers_found{};
    std::vector<std::string> modules;
    struct Issue { std::string code; std::string message; std::string subject; };
    std::vector<Issue> diagnostics;
    std::vector<DeviceInfo> devices;
    std::vector<std::string> errors;
};

struct StreamConfig {
    std::string device_args;
    double sample_rate{8000000.0};
    double center_frequency{2402000000.0};
    double bandwidth{2000000.0};
    double gain{30.0};
    std::string antenna;
    std::size_t channel{};
};

struct StreamMetrics {
    uint64_t samples{};
    uint64_t overflow{};
    uint64_t underflow{};
    uint64_t timeout{};
    uint64_t dropped{};
    uint64_t unsupported{};
    uint64_t device_lost{};
    uint64_t reconnects{};
    int64_t last_timestamp_ns{};
    int last_flags{};
    std::string last_error;
};

struct StreamIoResult { std::size_t count{}; int flags{}; int64_t timestamp_ns{}; };

} // namespace ofdm::radio
