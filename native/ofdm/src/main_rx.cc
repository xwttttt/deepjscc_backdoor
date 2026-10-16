#include "interface/zmq_interface.h"
#include "radio/soapy_radio.h"
#include "rx/rx_pipeline.h"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::atomic_bool running{true};
void signal_handler(int) { running = false; }

struct Options {
    std::string endpoint{"tcp://127.0.0.1:5556"};
    std::optional<std::string> device_args;
    std::optional<std::string> file_in;
    ofdm::RadioConfig radio;
    bool mmse{true};
    bool no_zmq{};
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
        else if (arg == "--file-in") options.file_in = value(i, argc, argv);
        else if (arg == "--frequency") options.radio.frequency_hz = std::stod(value(i, argc, argv));
        else if (arg == "--gain") options.radio.gain_db = std::stod(value(i, argc, argv));
        else if (arg == "--antenna") options.radio.antenna = value(i, argc, argv);
        else if (arg == "--channel") options.radio.channel = std::stoul(value(i, argc, argv));
        else if (arg == "--equalizer") {
            const auto mode = value(i, argc, argv);
            if (mode != "mmse" && mode != "zf") throw std::runtime_error("equalizer must be mmse or zf");
            options.mmse = mode == "mmse";
        } else if (arg == "--no-zmq") options.no_zmq = true;
        else if (arg == "--once") options.once = true;
        else if (arg == "--help") {
            std::cout
                << "Usage: ofdm-rx-service [options]\n"
                << "  --endpoint URI       ZeroMQ PUSH bind endpoint\n"
                << "  --device ARGS        SoapySDR device arguments\n"
                << "  --file-in PATH       read CF32 capture instead of SDR ('-' for stdin)\n"
                << "  --frequency HZ       center frequency (default 2437000000)\n"
                << "  --gain DB            RX gain (default 0)\n"
                << "  --antenna NAME       RX antenna\n"
                << "  --channel INDEX      SDR channel (default 0)\n"
                << "  --equalizer MODE     mmse (default) or zf\n"
                << "  --no-zmq             print decoded results without publishing\n"
                << "  --once               exit after one decoded burst\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.device_args.has_value() == options.file_in.has_value()) {
        throw std::runtime_error("select exactly one of --device or --file-in");
    }
    return options;
}

std::string result_json(const ofdm::RxMetadata& meta, const char* equalizer) {
    char buffer[768];
    std::snprintf(buffer, sizeof(buffer),
        "{\"version\":1,\"burst_id\":%u,\"format\":\"cf32le\",\"original_length\":%u,"
        "\"num_data_symbols\":%d,\"cfo_hz\":%.3f,\"rssi\":%.8f,\"snr_db\":%.3f,"
        "\"noise_var\":%.9f,\"csi_quality_db\":%.3f,\"equalizer\":\"%s\",\"crc_ok\":true,\"sync_offset\":%d}",
        meta.burst_id, meta.original_length, meta.num_data_symbols, meta.cfo_hz, meta.rssi,
        meta.snr_db, meta.noise_var, meta.csi_quality_db, equalizer, meta.sync_offset);
    return buffer;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        std::unique_ptr<ofdm::SoapyRxRadio> radio;
        std::ifstream input_file;
        std::istream* input{};
        if (options.device_args) {
            auto config = options.radio;
            config.device_args = *options.device_args;
            radio = std::make_unique<ofdm::SoapyRxRadio>(config);
        } else if (*options.file_in == "-") {
            input = &std::cin;
        } else {
            input_file.open(*options.file_in, std::ios::binary);
            if (!input_file) throw std::runtime_error("cannot open input file: " + *options.file_in);
            input = &input_file;
        }

        std::unique_ptr<ofdm::ZmqRxInterface> output;
        if (!options.no_zmq) output = std::make_unique<ofdm::ZmqRxInterface>(options.endpoint);
        ofdm::RxPipeline pipeline;
        pipeline.set_mmse(options.mmse);

        std::vector<cf32> chunk(65536);
        std::cerr << "[RX] source=" << (radio ? "SDR" : "file")
                  << " output=" << (output ? options.endpoint : "console") << '\n';

        while (running) {
            std::size_t count{};
            if (radio) {
                count = radio->read(chunk.data(), chunk.size());
                if (count == 0) continue;
            } else {
                input->read(reinterpret_cast<char*>(chunk.data()),
                            static_cast<std::streamsize>(chunk.size() * sizeof(cf32)));
                const auto bytes = input->gcount();
                if (bytes % static_cast<std::streamsize>(sizeof(cf32)) != 0) {
                    throw std::runtime_error("CF32 input ends with a partial complex sample");
                }
                count = static_cast<std::size_t>(bytes / sizeof(cf32));
                if (count == 0) break;
            }
            auto results = pipeline.process(chunk.data(), static_cast<int>(count));
            for (const auto& result : results) {
                const auto json = result_json(result.meta, options.mmse ? "mmse" : "zf");
                if (output) output->send(json, result.csi, result.data.data(), result.data.size());
                std::cout << json << '\n';
                if (options.once) running = false;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ofdm-rx-service: " << error.what() << '\n';
        return 1;
    }
}
