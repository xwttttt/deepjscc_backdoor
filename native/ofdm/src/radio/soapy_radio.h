#pragma once

#include "dsp/ofdm_core.h"

#include <cstddef>
#include <memory>
#include <string>

namespace ofdm {

struct RadioConfig {
    std::string device_args;
    double frequency_hz{2.437e9};
    double sample_rate{SAMPLE_RATE};
    double bandwidth_hz{20e6};
    double gain_db{0.0};
    std::string antenna;
    std::size_t channel{0};
};

class SoapyTxRadio {
public:
    explicit SoapyTxRadio(const RadioConfig& config);
    ~SoapyTxRadio();
    SoapyTxRadio(const SoapyTxRadio&) = delete;
    SoapyTxRadio& operator=(const SoapyTxRadio&) = delete;

    void write_burst(const cf32* samples, std::size_t count);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class SoapyRxRadio {
public:
    explicit SoapyRxRadio(const RadioConfig& config);
    ~SoapyRxRadio();
    SoapyRxRadio(const SoapyRxRadio&) = delete;
    SoapyRxRadio& operator=(const SoapyRxRadio&) = delete;

    // Returns samples read, zero for timeout, and throws on a device error.
    std::size_t read(cf32* samples, std::size_t capacity, long timeout_us = 100000);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One device handle with simultaneous RX and TX streams. This is the product
// path for full-duplex radios such as the B210 and avoids two processes
// competing for exclusive ownership of the same USB device.
class SoapyDuplexRadio {
public:
    SoapyDuplexRadio(const RadioConfig& rx_config, const RadioConfig& tx_config);
    ~SoapyDuplexRadio();
    SoapyDuplexRadio(const SoapyDuplexRadio&) = delete;
    SoapyDuplexRadio& operator=(const SoapyDuplexRadio&) = delete;

    void write_burst(const cf32* samples, std::size_t count);
    std::size_t read(cf32* samples, std::size_t capacity, long timeout_us = 100000);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ofdm
