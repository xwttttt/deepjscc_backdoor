#include "radio/device_manager.hpp"
#include <nlohmann/json.hpp>
#ifdef OFDM_HAVE_SOAPY
#include <SoapySDR/Device.hpp>
#include <SoapySDR/Modules.hpp>
#include <SoapySDR/Registry.hpp>
#endif
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;

Json ranges_json(const std::vector<ofdm::radio::Range>& ranges) {
    Json result = Json::array();
    for (const auto& range : ranges) result.push_back({{"minimum", range.minimum}, {"maximum", range.maximum}, {"step", range.step}});
    return result;
}

Json channels_json(const std::vector<ofdm::radio::RadioChannelInfo>& channels) {
    Json result = Json::array();
    for (const auto& channel : channels) {
        result.push_back({
            {"index", channel.index},
            {"label", channel.label},
            {"antennas", channel.antennas},
            {"sample_rates", ranges_json(channel.sample_rates)},
            {"frequencies", ranges_json(channel.frequencies)},
            {"gains", ranges_json(channel.gains)},
            {"bandwidths", ranges_json(channel.bandwidths)},
            {"full_duplex", channel.full_duplex},
        });
    }
    return result;
}

Json device_json(const ofdm::radio::DeviceInfo& device) {
    return {
        {"id", device.id},
        {"driver", device.driver},
        {"label", device.label},
        {"serial", device.serial},
        {"device_args", device.device_args},
        {"rx", device.rx},
        {"tx", device.tx},
        {"full_duplex", device.full_duplex},
        {"rx_channels", device.rx_channels},
        {"tx_channels", device.tx_channels},
        {"hardware_time", device.hardware_time},
        {"rx_antennas", device.rx_antennas},
        {"tx_antennas", device.tx_antennas},
        {"rx_channel_info", channels_json(device.rx_channel_info)},
        {"tx_channel_info", channels_json(device.tx_channel_info)},
        {"rx_sample_rates", ranges_json(device.rx_sample_rates)},
        {"tx_sample_rates", ranges_json(device.tx_sample_rates)},
        {"rx_frequencies", ranges_json(device.rx_frequencies)},
        {"tx_frequencies", ranges_json(device.tx_frequencies)},
        {"rx_gains", ranges_json(device.rx_gains)},
        {"tx_gains", ranges_json(device.tx_gains)},
        {"rx_bandwidths", ranges_json(device.rx_bandwidths)},
        {"tx_bandwidths", ranges_json(device.tx_bandwidths)},
    };
}

Json catalog_json(const ofdm::radio::DeviceDiagnostics& diagnostics) {
    Json devices = Json::array();
    for (const auto& device : diagnostics.devices) devices.push_back(device_json(device));
    Json issues = Json::array();
    for (const auto& issue : diagnostics.diagnostics) issues.push_back({{"code", issue.code}, {"message", issue.message}, {"subject", issue.subject}});
    return {
        {"catalog_version", 1},
        {"soapy_available", diagnostics.soapy_available},
        {"hardware_build", diagnostics.soapy_available},
        {"modules_found", diagnostics.modules_found},
        {"drivers_found", diagnostics.drivers_found},
        {"discovery_pending", false},
        {"modules", diagnostics.modules},
        {"devices", std::move(devices)},
        {"diagnostics", std::move(issues)},
        {"errors", diagnostics.errors},
    };
}

Json factories_json() {
#ifdef OFDM_HAVE_SOAPY
    const auto modules = SoapySDR::listModules();
    SoapySDR::loadModules();
    Json drivers = Json::array();
    Json diagnostics = Json::array();
    for (const auto& [driver, find] : SoapySDR::Registry::listFindFunctions()) {
        static_cast<void>(find);
        if (ofdm::radio::is_radio_driver_name(driver)) drivers.push_back(driver);
        else diagnostics.push_back({{"code", "non_rf_driver_excluded"}, {"message", "Non-RF SoapySDR driver was excluded from the global catalog"}, {"subject", driver}});
    }
    return {{"catalog_version", 1}, {"soapy_available", true}, {"hardware_build", true}, {"modules_found", !modules.empty()}, {"modules", modules}, {"drivers", std::move(drivers)}, {"diagnostics", std::move(diagnostics)}, {"errors", Json::array()}};
#else
    return {{"catalog_version", 1}, {"soapy_available", false}, {"hardware_build", false}, {"modules_found", false}, {"modules", Json::array()}, {"drivers", Json::array()}, {"diagnostics", Json::array({Json{{"code", "core_unavailable"}, {"message", "SoapySDR core was not available at build time"}, {"subject", "SoapySDR"}}})}, {"errors", Json::array()}};
#endif
}

/// Convert a JSON object to SoapySDR kwargs format: key1=val1, key2=val2
static std::string json_to_kwargs(const Json& obj) {
    std::string result;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        if (it.value().is_string()) {
            if (!result.empty()) result += ", ";
            result += it.key() + "=" + it.value().get<std::string>();
        }
    }
    return result;
}

/// Fast enumeration: list attached devices without opening them.
/// Returns basic metadata (driver, label, serial, device_args) only.
/// device_args is in SoapySDR kwargs format for direct use with Device::make().
Json list_json() {
#ifdef OFDM_HAVE_SOAPY
    SoapySDR::loadModules();
    auto results = SoapySDR::Device::enumerate();
    Json devices = Json::array();
    for (const auto& args : results) {
        // Skip non-RF drivers (audio, null, etc.)
        auto it = args.find("driver");
        if (it == args.end()) continue;
        if (!ofdm::radio::is_radio_driver_name(it->second)) continue;
        Json dev;
        for (const auto& [key, val] : args) dev[key] = val;
        dev["device_args"] = json_to_kwargs(dev);
        devices.push_back(std::move(dev));
    }
    return {{"catalog_version", 1}, {"soapy_available", true}, {"hardware_build", true}, {"devices", std::move(devices)}, {"diagnostics", Json::array()}, {"errors", Json::array()}};
#else
    return {{"catalog_version", 1}, {"soapy_available", false}, {"hardware_build", false}, {"devices", Json::array()}, {"diagnostics", Json::array()}, {"errors", Json::array()}};
#endif
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: ofdm-device-probe --list | --factories | --driver DRIVER | --device JSON_ARGS\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string(argv[1]) == "--list") std::cout << list_json().dump() << '\n';
        else if (argc == 2 && std::string(argv[1]) == "--factories") std::cout << factories_json().dump() << '\n';
        else if (argc == 3 && std::string(argv[1]) == "--driver") std::cout << catalog_json(ofdm::radio::DeviceManager{argv[2]}.discover()).dump() << '\n';
        else if (argc == 3 && std::string(argv[1]) == "--device") {
            // device_args is in SoapySDR kwargs format (driver=uhd, serial=32FCCE4)
            // Parse kwargs to find driver name
            SoapySDR::Kwargs kwargs = SoapySDR::KwargsFromString(argv[2]);
            auto it = kwargs.find("driver");
            if (it == kwargs.end() || it->second.empty()) throw std::runtime_error("device_args must contain 'driver'");
            std::cout << catalog_json(ofdm::radio::DeviceManager{it->second}.discover()).dump() << '\n';
        }
        else throw std::runtime_error("expected --list, --factories, --driver DRIVER, or --device JSON_ARGS");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ofdm-device-probe: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
