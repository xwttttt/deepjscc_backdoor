#include "radio/device_manager.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>

#ifdef OFDM_HAVE_SOAPY
#include <SoapySDR/Device.hpp>
#include <SoapySDR/Modules.hpp>
#include <SoapySDR/Registry.hpp>
#include <SoapySDR/Types.hpp>
#endif

namespace ofdm::radio {

bool is_radio_driver_name(const std::string& name) {
    std::string driver = name;
    std::transform(driver.begin(), driver.end(), driver.begin(), [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (driver == "null") return false;
    for (const auto* non_radio : {"audio", "alsa", "pulse", "portaudio", "sound"}) {
        if (driver.find(non_radio) != std::string::npos) return false;
    }
    return true;
}

namespace {

#ifdef OFDM_HAVE_SOAPY
std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string channel_label(
    SoapySDR::Device& device,
    const DeviceInfo& device_info,
    const std::string& product,
    int direction,
    std::size_t channel
) {
    const auto identity = lowercase(product + " " + device_info.label);
    if (identity.find("b210") != std::string::npos && channel < 2) {
        return std::string(1, static_cast<char>('A' + channel));
    }
    try {
        const auto details = device.getChannelInfo(direction, channel);
        for (const auto* key : {"label", "name", "frontend", "frontend_name", "rf_frontend"}) {
            const auto found = details.find(key);
            if (found != details.end() && !found->second.empty()) return found->second;
        }
    } catch (...) {
        // Channel metadata is optional in SoapySDR. Capability probing continues below.
    }
    return std::string(direction == SOAPY_SDR_RX ? "RX " : "TX ") + std::to_string(channel);
}

RadioChannelInfo probe_channel(
    SoapySDR::Device& device,
    const DeviceInfo& device_info,
    const std::string& product,
    int direction,
    std::size_t channel,
    DeviceDiagnostics& diagnostics
) {
    RadioChannelInfo info;
    info.index = channel;
    info.label = channel_label(device, device_info, product, direction, channel);
    const auto direction_name = direction == SOAPY_SDR_RX ? "RX" : "TX";
    const auto subject = device_info.id + " " + direction_name + " channel " + std::to_string(channel);
    const auto probe = [&](const char* capability, const auto& operation) {
        try {
            operation();
        } catch (const std::exception& error) {
            diagnostics.diagnostics.push_back({"channel_probe_error", std::string(capability) + ": " + error.what(), subject});
        }
    };
    probe("antennas", [&] { info.antennas = device.listAntennas(direction, channel); });
    probe("sample rates", [&] {
        for (const auto& range : device.getSampleRateRange(direction, channel)) info.sample_rates.push_back({range.minimum(), range.maximum(), range.step()});
    });
    probe("frequencies", [&] {
        for (const auto& range : device.getFrequencyRange(direction, channel)) info.frequencies.push_back({range.minimum(), range.maximum(), range.step()});
    });
    probe("gain", [&] {
        const auto range = device.getGainRange(direction, channel);
        info.gains.push_back({range.minimum(), range.maximum(), range.step()});
    });
    probe("bandwidths", [&] {
        for (const auto& range : device.getBandwidthRange(direction, channel)) info.bandwidths.push_back({range.minimum(), range.maximum(), range.step()});
    });
    probe("duplex", [&] { info.full_duplex = device.getFullDuplex(direction, channel); });
    return info;
}
#endif

class SystemDeviceDiscoveryBackend final : public DeviceDiscoveryBackend {
public:
explicit SystemDeviceDiscoveryBackend(std::string driver = {}) : driver_(std::move(driver)) {}
DeviceDiagnostics discover() const override {
    DeviceDiagnostics result;
#ifdef OFDM_HAVE_SOAPY
    result.soapy_available = true;
    try {
        result.modules = SoapySDR::listModules();
        result.modules_found = !result.modules.empty();
        if (!result.modules_found) result.diagnostics.push_back({"no_modules", "SoapySDR found no loadable modules; install a device plugin", "SoapySDR"});
        for (const auto& module : result.modules) {
            try { static_cast<void>(SoapySDR::getModuleVersion(module)); }
            catch (const std::exception& error) { result.diagnostics.push_back({"module_load_error", error.what(), module}); }
        }
        std::set<std::string> drivers;
        std::vector<SoapySDR::Kwargs> enumerated;
        SoapySDR::loadModules();
        for (const auto& [driver, find] : SoapySDR::Registry::listFindFunctions()) {
            static_cast<void>(find);
            if (!driver_.empty() && driver != driver_) continue;
            if (!is_radio_driver_name(driver)) {
                result.diagnostics.push_back({"non_rf_driver_excluded", "Non-RF SoapySDR driver was not enumerated", driver});
                continue;
            }
            drivers.insert(driver);
            try {
                const auto devices = SoapySDR::Device::enumerate(SoapySDR::Kwargs{{"driver", driver}});
                enumerated.insert(enumerated.end(), devices.begin(), devices.end());
            } catch (const std::exception& error) {
                result.diagnostics.push_back({"driver_enumeration_error", error.what(), driver});
            }
        }
        std::set<std::string> probed;
        for (const auto& args : enumerated) {
            DeviceInfo info;
            auto get = [&args](const std::string& key) {
                const auto it = args.find(key);
                return it == args.end() ? std::string{} : it->second;
            };
            info.driver = get("driver");
            if (!info.driver.empty()) drivers.insert(info.driver);
            info.label = get("label");
            info.serial = get("serial");
            info.device_args = SoapySDR::KwargsToString(args);
            if (!probed.insert(info.device_args).second) continue;
            info.id = info.driver + ":" + (info.serial.empty() ? info.label : info.serial);
            if (args.size() == 1 && info.label.empty() && info.serial.empty()) {
                result.diagnostics.push_back({"unattached_device_template_excluded", "Driver returned no attached hardware identity or address", info.driver});
                continue;
            }
            try {
                using DeviceHandle = std::unique_ptr<SoapySDR::Device, void (*)(SoapySDR::Device*)>;
                DeviceHandle device(SoapySDR::Device::make(args), [](SoapySDR::Device* value) { if (value) SoapySDR::Device::unmake(value); });
                if (!device) {
                    result.errors.push_back("SoapySDR returned no device for " + info.id);
                    result.diagnostics.push_back({"device_open_error", "SoapySDR returned no device", info.id});
                    continue;
                }
                info.rx_channels = device->getNumChannels(SOAPY_SDR_RX);
                info.tx_channels = device->getNumChannels(SOAPY_SDR_TX);
                info.rx = info.rx_channels > 0;
                info.tx = info.tx_channels > 0;
                if (info.rx && info.tx) {
                    try {
                        info.full_duplex =
                            device->getFullDuplex(SOAPY_SDR_RX, 0) &&
                            device->getFullDuplex(SOAPY_SDR_TX, 0);
                    } catch (const std::exception& error) {
                        info.full_duplex = false;
                        result.diagnostics.push_back({"full_duplex_probe_error", error.what(), info.id});
                    }
                }
                info.hardware_time = device->hasHardwareTime();
                const auto product = get("product");
                for (std::size_t channel = 0; channel < info.rx_channels; ++channel) {
                    info.rx_channel_info.push_back(probe_channel(*device, info, product, SOAPY_SDR_RX, channel, result));
                }
                for (std::size_t channel = 0; channel < info.tx_channels; ++channel) {
                    info.tx_channel_info.push_back(probe_channel(*device, info, product, SOAPY_SDR_TX, channel, result));
                }
                if (info.rx) {
                    info.rx_antennas = info.rx_channel_info.front().antennas;
                    info.antennas = info.rx_antennas;
                    info.rx_sample_rates = info.rx_channel_info.front().sample_rates;
                    info.rx_frequencies = info.rx_channel_info.front().frequencies;
                    info.rx_gains = info.rx_channel_info.front().gains;
                    info.rx_bandwidths = info.rx_channel_info.front().bandwidths;
                    info.sample_rates = info.rx_sample_rates;
                    info.frequencies = info.rx_frequencies;
                    info.gains = info.rx_gains;
                }
                if (info.tx) {
                    info.tx_antennas = info.tx_channel_info.front().antennas;
                    info.tx_sample_rates = info.tx_channel_info.front().sample_rates;
                    info.tx_frequencies = info.tx_channel_info.front().frequencies;
                    info.tx_gains = info.tx_channel_info.front().gains;
                    info.tx_bandwidths = info.tx_channel_info.front().bandwidths;
                }
                result.devices.push_back(std::move(info));
            } catch (const std::exception& error) {
                result.errors.push_back(info.id + ": " + error.what());
                result.diagnostics.push_back({"device_probe_error", error.what(), info.id});
            }
        }
        result.drivers_found = !drivers.empty();
        if (!result.drivers_found) result.diagnostics.push_back({"no_drivers", "No SoapySDR device driver reported an attached device", "SoapySDR"});
    } catch (const std::exception& error) {
        result.errors.push_back(error.what());
        result.diagnostics.push_back({"discovery_error", error.what(), "SoapySDR"});
    }
#else
    result.errors.push_back("SoapySDR core was not available at build time; file replay and simulation remain available");
    result.diagnostics.push_back({"core_unavailable", result.errors.back(), "SoapySDR"});
#endif
    return result;
}

private:
    std::string driver_;
};

}

std::unique_ptr<DeviceDiscoveryBackend> make_system_device_discovery_backend() { return std::make_unique<SystemDeviceDiscoveryBackend>(); }

bool is_radio_device(const DeviceInfo& device) {
    if (!is_radio_driver_name(device.driver)) return false;
    const auto has_tuning_range = [](const std::vector<Range>& ranges) {
        return std::any_of(ranges.begin(), ranges.end(), [](const Range& range) {
            return std::isfinite(range.minimum) && std::isfinite(range.maximum) && range.minimum >= 0.0 && range.maximum >= range.minimum && range.maximum > 0.0;
        });
    };
    return (device.rx && has_tuning_range(device.rx_frequencies)) || (device.tx && has_tuning_range(device.tx_frequencies));
}

DeviceManager::DeviceManager() : DeviceManager(make_system_device_discovery_backend()) {}
DeviceManager::DeviceManager(const std::string& driver) : DeviceManager(std::make_unique<SystemDeviceDiscoveryBackend>(driver)) {}
DeviceManager::DeviceManager(std::unique_ptr<DeviceDiscoveryBackend> backend) : backend_(std::move(backend)) { if (!backend_) throw std::invalid_argument("device discovery backend is required"); }
DeviceDiagnostics DeviceManager::discover() const {
    auto result = backend_->discover();
    const auto before = result.devices.size();
    result.devices.erase(std::remove_if(result.devices.begin(), result.devices.end(), [](const DeviceInfo& device) { return !is_radio_device(device); }), result.devices.end());
    const auto excluded = before - result.devices.size();
    if (excluded) result.diagnostics.push_back({"non_rf_endpoints_excluded", std::to_string(excluded) + " non-RF SoapySDR endpoint(s) were excluded", "device catalog"});
    return result;
}

}
