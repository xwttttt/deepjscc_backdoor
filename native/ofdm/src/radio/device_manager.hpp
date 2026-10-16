#pragma once

#include "radio/types.hpp"

#include <memory>

namespace ofdm::radio {

class DeviceDiscoveryBackend {
public:
    virtual ~DeviceDiscoveryBackend() = default;
    virtual DeviceDiagnostics discover() const = 0;
};

std::unique_ptr<DeviceDiscoveryBackend> make_system_device_discovery_backend();
bool is_radio_driver_name(const std::string& driver);
bool is_radio_device(const DeviceInfo& device);

class DeviceManager {
public:
    DeviceManager();
    explicit DeviceManager(const std::string& driver);
    explicit DeviceManager(std::unique_ptr<DeviceDiscoveryBackend> backend);
    DeviceDiagnostics discover() const;

private:
    std::unique_ptr<DeviceDiscoveryBackend> backend_;
};

} // namespace ofdm::radio
