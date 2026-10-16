#include "radio/soapy_radio.h"

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Errors.hpp>
#include <SoapySDR/Formats.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ofdm {
namespace {

void configure(SoapySDR::Device* device, int direction, const RadioConfig& config) {
    device->setSampleRate(direction, config.channel, config.sample_rate);
    device->setFrequency(direction, config.channel, config.frequency_hz);
    device->setBandwidth(direction, config.channel, config.bandwidth_hz);
    device->setGain(direction, config.channel, config.gain_db);
    if (!config.antenna.empty()) device->setAntenna(direction, config.channel, config.antenna);

    const auto actual_rate = device->getSampleRate(direction, config.channel);
    if (std::abs(actual_rate - config.sample_rate) > 1.0) {
        throw std::runtime_error("SDR did not accept the required 20 Msps sample rate");
    }
}

std::runtime_error stream_error(const char* operation, int code) {
    return std::runtime_error(std::string(operation) + ": " + SoapySDR::errToStr(code));
}

} // namespace

struct SoapyTxRadio::Impl {
    SoapySDR::Device* device{};
    SoapySDR::Stream* stream{};

    explicit Impl(const RadioConfig& config) {
        device = SoapySDR::Device::make(SoapySDR::KwargsFromString(config.device_args));
        if (!device) throw std::runtime_error("unable to create TX SDR: " + config.device_args);
        try {
            configure(device, SOAPY_SDR_TX, config);
            stream = device->setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, {config.channel});
            const int status = device->activateStream(stream);
            if (status != 0) throw stream_error("activate TX stream", status);
        } catch (...) {
            if (stream) device->closeStream(stream);
            SoapySDR::Device::unmake(device);
            throw;
        }
    }

    ~Impl() {
        if (device && stream) {
            device->deactivateStream(stream);
            device->closeStream(stream);
        }
        if (device) SoapySDR::Device::unmake(device);
    }
};

SoapyTxRadio::SoapyTxRadio(const RadioConfig& config) : impl_(std::make_unique<Impl>(config)) {}
SoapyTxRadio::~SoapyTxRadio() = default;

void SoapyTxRadio::write_burst(const cf32* samples, std::size_t count) {
    std::size_t offset = 0;
    const std::size_t mtu = std::max<std::size_t>(1, impl_->device->getStreamMTU(impl_->stream));
    while (offset < count) {
        const auto chunk = std::min(mtu, count - offset);
        const void* buffers[] = {samples + offset};
        int flags = offset + chunk == count ? SOAPY_SDR_END_BURST : 0;
        const int written = impl_->device->writeStream(impl_->stream, buffers, chunk, flags, 0, 1000000);
        if (written == SOAPY_SDR_TIMEOUT) continue;
        if (written < 0) throw stream_error("write TX stream", written);
        if (written == 0) throw std::runtime_error("write TX stream made no progress");
        offset += static_cast<std::size_t>(written);
    }
}

struct SoapyRxRadio::Impl {
    SoapySDR::Device* device{};
    SoapySDR::Stream* stream{};

    explicit Impl(const RadioConfig& config) {
        device = SoapySDR::Device::make(SoapySDR::KwargsFromString(config.device_args));
        if (!device) throw std::runtime_error("unable to create RX SDR: " + config.device_args);
        try {
            configure(device, SOAPY_SDR_RX, config);
            stream = device->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {config.channel});
            const int status = device->activateStream(stream);
            if (status != 0) throw stream_error("activate RX stream", status);
        } catch (...) {
            if (stream) device->closeStream(stream);
            SoapySDR::Device::unmake(device);
            throw;
        }
    }

    ~Impl() {
        if (device && stream) {
            device->deactivateStream(stream);
            device->closeStream(stream);
        }
        if (device) SoapySDR::Device::unmake(device);
    }
};

SoapyRxRadio::SoapyRxRadio(const RadioConfig& config) : impl_(std::make_unique<Impl>(config)) {}
SoapyRxRadio::~SoapyRxRadio() = default;

std::size_t SoapyRxRadio::read(cf32* samples, std::size_t capacity, long timeout_us) {
    void* buffers[] = {samples};
    int flags = 0;
    long long time_ns = 0;
    const int received = impl_->device->readStream(impl_->stream, buffers, capacity, flags, time_ns, timeout_us);
    if (received == SOAPY_SDR_TIMEOUT) return 0;
    // An RX overflow loses samples and therefore invalidates the current
    // acquisition, but it is recoverable; let the caller continue and allow
    // the synchronizer's bounded buffer to reacquire on a later burst.
    if (received == SOAPY_SDR_OVERFLOW) return 0;
    if (received < 0) throw stream_error("read RX stream", received);
    return static_cast<std::size_t>(received);
}

struct SoapyDuplexRadio::Impl {
    SoapySDR::Device* device{};
    SoapySDR::Stream* rx_stream{};
    SoapySDR::Stream* tx_stream{};

    Impl(const RadioConfig& rx_config, const RadioConfig& tx_config) {
        if (rx_config.device_args != tx_config.device_args) {
            throw std::invalid_argument("duplex mode requires one shared SoapySDR device");
        }
        device = SoapySDR::Device::make(SoapySDR::KwargsFromString(rx_config.device_args));
        if (!device) throw std::runtime_error("unable to create duplex SDR: " + rx_config.device_args);
        try {
            configure(device, SOAPY_SDR_RX, rx_config);
            configure(device, SOAPY_SDR_TX, tx_config);
            rx_stream = device->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {rx_config.channel});
            tx_stream = device->setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, {tx_config.channel});
            int status = device->activateStream(rx_stream);
            if (status != 0) throw stream_error("activate duplex RX stream", status);
            status = device->activateStream(tx_stream);
            if (status != 0) throw stream_error("activate duplex TX stream", status);
        } catch (...) {
            if (tx_stream) device->closeStream(tx_stream);
            if (rx_stream) device->closeStream(rx_stream);
            SoapySDR::Device::unmake(device);
            throw;
        }
    }

    ~Impl() {
        if (device && tx_stream) {
            device->deactivateStream(tx_stream);
            device->closeStream(tx_stream);
        }
        if (device && rx_stream) {
            device->deactivateStream(rx_stream);
            device->closeStream(rx_stream);
        }
        if (device) SoapySDR::Device::unmake(device);
    }
};

SoapyDuplexRadio::SoapyDuplexRadio(const RadioConfig& rx_config, const RadioConfig& tx_config)
    : impl_(std::make_unique<Impl>(rx_config, tx_config)) {}
SoapyDuplexRadio::~SoapyDuplexRadio() = default;

void SoapyDuplexRadio::write_burst(const cf32* samples, std::size_t count) {
    std::size_t offset = 0;
    const auto mtu = std::max<std::size_t>(1, impl_->device->getStreamMTU(impl_->tx_stream));
    while (offset < count) {
        const auto chunk = std::min(mtu, count - offset);
        const void* buffers[] = {samples + offset};
        int flags = offset + chunk == count ? SOAPY_SDR_END_BURST : 0;
        const int written = impl_->device->writeStream(impl_->tx_stream, buffers, chunk, flags, 0, 1000000);
        if (written == SOAPY_SDR_TIMEOUT) continue;
        if (written < 0) throw stream_error("write duplex TX stream", written);
        if (written == 0) throw std::runtime_error("write duplex TX stream made no progress");
        offset += static_cast<std::size_t>(written);
    }
}

std::size_t SoapyDuplexRadio::read(cf32* samples, std::size_t capacity, long timeout_us) {
    void* buffers[] = {samples};
    int flags = 0;
    long long time_ns = 0;
    const int received = impl_->device->readStream(impl_->rx_stream, buffers, capacity, flags, time_ns, timeout_us);
    if (received == SOAPY_SDR_TIMEOUT || received == SOAPY_SDR_OVERFLOW) return 0;
    if (received < 0) throw stream_error("read duplex RX stream", received);
    return static_cast<std::size_t>(received);
}

} // namespace ofdm
