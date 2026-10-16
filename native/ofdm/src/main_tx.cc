#include "interface/zmq_interface.h"
#include "radio/soapy_radio.h"
#include "tx/tx_pipeline.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

std::atomic_bool running{true};
void signal_handler(int) { running = false; }

struct Options {
    std::string endpoint{"tcp://127.0.0.1:5555"};
    std::optional<std::string> device_args;
    std::optional<std::string> file_out;
    ofdm::RadioConfig radio;
    bool dry_run{};
    bool once{};
};

std::string value(int& index, int argc, char** argv) {
    if (++index >= argc) throw std::runtime_error(std::string("missing value for ") + argv[index - 1]);
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--endpoint") options.endpoint = value(i, argc, argv);
        else if (arg == "--device") options.device_args = value(i, argc, argv);
        else if (arg == "--file-out") options.file_out = value(i, argc, argv);
        else if (arg == "--frequency") options.radio.frequency_hz = std::stod(value(i, argc, argv));
        else if (arg == "--gain") options.radio.gain_db = std::stod(value(i, argc, argv));
        else if (arg == "--antenna") options.radio.antenna = value(i, argc, argv);
        else if (arg == "--channel") options.radio.channel = std::stoul(value(i, argc, argv));
        else if (arg == "--arm-seconds") static_cast<void>(value(i, argc, argv)); // deprecated compatibility
        else if (arg == "--dry-run") options.dry_run = true;
        else if (arg == "--once") options.once = true;
        else if (arg == "--help") {
            std::cout
                << "Usage: ofdm-tx-service [options]\n"
                << "  --endpoint URI       ZeroMQ PULL bind endpoint\n"
                << "  --device ARGS        SoapySDR device arguments\n"
                << "  --frequency HZ       center frequency (default 2437000000)\n"
                << "  --gain DB            TX gain (default 0)\n"
                << "  --antenna NAME       TX antenna\n"
                << "  --channel INDEX      SDR channel (default 0)\n"
                << "  --arm-seconds N      deprecated and ignored\n"
                << "  --file-out PATH      write generated CF32 bursts instead of RF\n"
                << "  --dry-run            build and validate bursts without output\n"
                << "  --once               exit after one accepted request\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    const int outputs = static_cast<int>(options.device_args.has_value()) +
                        static_cast<int>(options.file_out.has_value()) + static_cast<int>(options.dry_run);
    if (outputs != 1) throw std::runtime_error("select exactly one of --device, --file-out, or --dry-run");
    return options;
}

uint32_t json_unsigned(const std::string& json, const std::string& key, uint32_t fallback) {
    const auto name = '"' + key + '"';
    auto pos = json.find(name);
    if (pos == std::string::npos || (pos = json.find(':', pos + name.size())) == std::string::npos) return fallback;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    std::size_t end{};
    try {
        const auto parsed = std::stoull(json.substr(pos), &end);
        if (end == 0 || parsed > UINT32_MAX) return fallback;
        return static_cast<uint32_t>(parsed);
    } catch (...) {
        return fallback;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        std::unique_ptr<ofdm::SoapyTxRadio> radio;
        if (options.device_args) {
            auto config = options.radio;
            config.device_args = *options.device_args;
            radio = std::make_unique<ofdm::SoapyTxRadio>(config);
        }
        std::ofstream output;
        if (options.file_out) {
            output.open(*options.file_out, std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("cannot open output file: " + *options.file_out);
        }

        ofdm::ZmqTxInterface input(options.endpoint);
        ofdm::TxPipeline pipeline;
        uint32_t next_id = 1;
        std::cerr << "[TX] input=" << options.endpoint
                  << (radio ? " output=SDR (enabled until stopped)" :
                      options.file_out ? " output=file" : " output=dry-run")
                  << '\n';

        while (running) {
            std::string metadata;
            std::vector<cf32> data;
            if (!input.receive(metadata, data)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            const uint32_t burst_id = json_unsigned(metadata, "burst_id", next_id++);
            auto burst = pipeline.build_burst(data.data(), data.size(), burst_id);
            if (radio) radio->write_burst(burst.samples.data(), burst.samples.size());
            if (output) output.write(reinterpret_cast<const char*>(burst.samples.data()),
                                     static_cast<std::streamsize>(burst.samples.size() * sizeof(cf32)));
            std::cerr << "[TX] burst=" << burst.header.burst_id << " values=" << data.size()
                      << " symbols=" << burst.header.num_data_symbols << " samples=" << burst.samples.size()
                      << " scale=" << burst.waveform_scale << '\n';
            if (options.once) break;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ofdm-tx-service: " << error.what() << '\n';
        return 1;
    }
}
