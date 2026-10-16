#include "interface/telemetry.h"
#include "interface/zmq_interface.h"
#include "radio/soapy_radio.h"
#include "rx/rx_pipeline.h"
#include "tx/tx_pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic_bool running{true};
void signal_handler(int) { running = false; }

struct Options {
    enum class Role { Rx, Tx, Duplex } role{Role::Duplex};
    std::string tx_endpoint{"tcp://127.0.0.1:5555"};
    std::string rx_endpoint{"tcp://127.0.0.1:5556"};
    std::optional<std::string> device_args;
    ofdm::RadioConfig rx;
    ofdm::RadioConfig tx;
    bool simulate{};
    bool demo_source{};
};

std::string value(int& index, int argc, char** argv) {
    if (++index >= argc) throw std::runtime_error(std::string("missing value for ") + argv[index - 1]);
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tx-endpoint") options.tx_endpoint = value(i, argc, argv);
        else if (arg == "--rx-endpoint") options.rx_endpoint = value(i, argc, argv);
        else if (arg == "--device") options.device_args = value(i, argc, argv);
        else if (arg == "--frequency") options.rx.frequency_hz = options.tx.frequency_hz = std::stod(value(i, argc, argv));
        else if (arg == "--rx-gain") options.rx.gain_db = std::stod(value(i, argc, argv));
        else if (arg == "--tx-gain") options.tx.gain_db = std::stod(value(i, argc, argv));
        else if (arg == "--rx-channel") options.rx.channel = std::stoul(value(i, argc, argv));
        else if (arg == "--tx-channel") options.tx.channel = std::stoul(value(i, argc, argv));
        else if (arg == "--rx-antenna") options.rx.antenna = value(i, argc, argv);
        else if (arg == "--tx-antenna") options.tx.antenna = value(i, argc, argv);
        else if (arg == "--arm-seconds") static_cast<void>(value(i, argc, argv)); // deprecated compatibility
        else if (arg == "--role") {
            const auto role = value(i, argc, argv);
            if (role == "rx") options.role = Options::Role::Rx;
            else if (role == "tx") options.role = Options::Role::Tx;
            else if (role == "duplex") options.role = Options::Role::Duplex;
            else throw std::runtime_error("--role must be rx, tx, or duplex");
        }
        else if (arg == "--simulate") options.simulate = true;
        else if (arg == "--demo-source") options.demo_source = true;
        else if (arg == "--help") {
            std::cout << "Usage: ofdm-workbench-service (--simulate | --device ARGS) [options]\n"
                      << "  --tx-endpoint URI --rx-endpoint URI\n"
                      << "  --frequency HZ --rx-gain DB --tx-gain DB\n"
                      << "  --rx-channel N --tx-channel N --rx-antenna NAME --tx-antenna NAME\n"
                      << "  --role rx|tx|duplex (hardware stream lifecycle)\n"
                      << "  --demo-source (paced deterministic CF32 source for simulation)\n"
                      << "  --arm-seconds N (deprecated and ignored)\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + arg);
    }
    if (options.simulate == options.device_args.has_value()) throw std::runtime_error("select exactly one of --simulate or --device");
    if (options.demo_source && !options.simulate) throw std::runtime_error("--demo-source requires --simulate");
    if (options.device_args) options.rx.device_args = options.tx.device_args = *options.device_args;
    return options;
}

uint32_t burst_id_from_json(const std::string& json, uint32_t fallback) {
    const std::string name{"\"burst_id\""};
    auto pos = json.find(name);
    if (pos == std::string::npos || (pos = json.find(':', pos + name.size())) == std::string::npos) return fallback;
    try {
        const auto parsed = std::stoull(json.substr(pos + 1));
        return parsed <= UINT32_MAX ? static_cast<uint32_t>(parsed) : fallback;
    } catch (...) { return fallback; }
}

std::string result_json(const ofdm::RxMetadata& meta) {
    char out[768];
    std::snprintf(out, sizeof(out),
        "{\"version\":1,\"burst_id\":%u,\"format\":\"cf32le\",\"original_length\":%u,"
        "\"num_data_symbols\":%d,\"cfo_hz\":%.3f,\"rssi\":%.8f,\"snr_db\":%.3f,"
        "\"noise_var\":%.9f,\"csi_quality_db\":%.3f,\"equalizer\":\"mmse\",\"crc_ok\":true,\"sync_offset\":%d}",
        meta.burst_id, meta.original_length, meta.num_data_symbols, meta.cfo_hz, meta.rssi,
        meta.snr_db, meta.noise_var, meta.csi_quality_db, meta.sync_offset);
    return out;
}

struct Counters {
    std::atomic<uint64_t> tx_bursts{};
    std::atomic<uint64_t> tx_samples{};
    std::atomic<uint64_t> rx_samples{};
    std::atomic<uint64_t> rx_bursts{};
    std::atomic<uint64_t> errors{};
    std::atomic<uint64_t> rx_drops{};
    std::atomic<uint64_t> rx_stf{};
    std::atomic<uint64_t> rx_ltf{};
    std::atomic<uint64_t> rx_csi_rejects{};
    std::atomic<uint64_t> rx_control_failures{};
    std::atomic<uint64_t> rx_header_failures{};
    std::atomic<float> rx_last_csi_db{-100.0f};
    std::atomic<float> rx_max_csi_db{-100.0f};
    std::atomic<float> rx_last_cfo_hz{};
};

struct RxSampleQueue {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::vector<cf32>> chunks;
    std::deque<std::vector<cf32>> free_chunks;
    // Marconi's Wi-Fi pipeline uses a 64-block bounded queue (about 8 MiB
    // at CF32/16k samples). Keep the same amount of burst-time headroom.
    std::size_t maximum_chunks{64};
    uint64_t discontinuities{};
    bool closed{};

    RxSampleQueue() {
        for (std::size_t index = 0; index < maximum_chunks + 4; ++index) {
            std::vector<cf32> block;
            block.reserve(16384);
            free_chunks.push_back(std::move(block));
        }
    }
};

void publish_status(ofdm::TelemetryWriter& telemetry, const Counters& counters,
                    bool simulation, bool tx_enabled, const std::string& detail = "") {
    char out[768];
    std::snprintf(out, sizeof(out),
        "{\"state\":\"%s\",\"mode\":\"%s\",\"armed\":%s,\"tx_enabled\":%s,\"tx_bursts\":%llu,"
        "\"tx_samples\":%llu,\"rx_bursts\":%llu,\"rx_samples\":%llu,\"errors\":%llu,\"rx_drops\":%llu,"
        "\"rx_stf\":%llu,\"rx_ltf\":%llu,\"rx_csi_rejects\":%llu,\"rx_control_failures\":%llu,\"rx_header_failures\":%llu,"
        "\"rx_last_csi_db\":%.2f,\"rx_max_csi_db\":%.2f,\"rx_last_cfo_hz\":%.2f,\"detail\":\"%s\"}",
        running ? "running" : "stopped", simulation ? "simulation" : "hardware",
        tx_enabled ? "true" : "false", tx_enabled ? "true" : "false",
        static_cast<unsigned long long>(counters.tx_bursts.load()),
        static_cast<unsigned long long>(counters.tx_samples.load()),
        static_cast<unsigned long long>(counters.rx_bursts.load()),
        static_cast<unsigned long long>(counters.rx_samples.load()),
        static_cast<unsigned long long>(counters.errors.load()),
        static_cast<unsigned long long>(counters.rx_drops.load()),
        static_cast<unsigned long long>(counters.rx_stf.load()),
        static_cast<unsigned long long>(counters.rx_ltf.load()),
        static_cast<unsigned long long>(counters.rx_csi_rejects.load()),
        static_cast<unsigned long long>(counters.rx_control_failures.load()),
        static_cast<unsigned long long>(counters.rx_header_failures.load()),
        static_cast<double>(counters.rx_last_csi_db.load()),
        static_cast<double>(counters.rx_max_csi_db.load()),
        static_cast<double>(counters.rx_last_cfo_hz.load()), detail.c_str());
    telemetry.json(ofdm::TelemetryKind::Status, out);
}

void publish_result(const ofdm::RxResult& result, ofdm::ZmqRxInterface* output,
                    ofdm::TelemetryWriter& telemetry, Counters& counters) {
    const auto json = result_json(result.meta);
    if (output) output->send(json, result.csi, result.data.data(), result.data.size());
    telemetry.iq(ofdm::TelemetryKind::Csi, result.csi, ofdm::FFT_SIZE, ofdm::FFT_SIZE);
    ++counters.rx_bursts;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        ofdm::TelemetryWriter telemetry;
        std::unique_ptr<ofdm::ZmqRxInterface> output;
        if (options.simulate || options.role != Options::Role::Tx) output = std::make_unique<ofdm::ZmqRxInterface>(options.rx_endpoint);
        Counters counters;

        std::unique_ptr<ofdm::SoapyDuplexRadio> radio;
        std::unique_ptr<ofdm::SoapyRxRadio> rx_radio;
        std::unique_ptr<ofdm::SoapyTxRadio> tx_radio;
        if (!options.simulate) {
            if (options.role == Options::Role::Duplex) radio = std::make_unique<ofdm::SoapyDuplexRadio>(options.rx, options.tx);
            else if (options.role == Options::Role::Rx) rx_radio = std::make_unique<ofdm::SoapyRxRadio>(options.rx);
            else tx_radio = std::make_unique<ofdm::SoapyTxRadio>(options.tx);
        }
        std::thread receiver;
        std::thread decoder;
        std::thread transmitter;
        std::shared_ptr<RxSampleQueue> rx_queue;
        if (radio || rx_radio) {
            rx_queue = std::make_shared<RxSampleQueue>();
            const auto queue = rx_queue;

            // Keep SDR I/O independent from FFT/synchronization.  The old
            // implementation called pipeline.process() in this same loop;
            // a slow decode then stopped readStream(), causing UHD/USB
            // overflows and lost bursts.
            receiver = std::thread([&, queue] {
                try {
                    constexpr std::size_t chunk_capacity = 16384;
                    std::vector<cf32> chunk(chunk_capacity);
                    auto next_plot = std::chrono::steady_clock::now();
                    while (running) {
                        const auto count = radio ? radio->read(chunk.data(), chunk.size()) : rx_radio->read(chunk.data(), chunk.size());
                        if (!count) continue;
                        counters.rx_samples.fetch_add(count);
                        const auto now = std::chrono::steady_clock::now();
                        if (now >= next_plot) {
                            telemetry.iq(ofdm::TelemetryKind::RxIq, chunk.data(), count);
                            next_plot = now + std::chrono::milliseconds(50);
                        }

                        std::vector<cf32> block;
                        {
                            std::lock_guard<std::mutex> lock(queue->mutex);
                            if (!queue->free_chunks.empty()) {
                                block = std::move(queue->free_chunks.front());
                                queue->free_chunks.pop_front();
                            }
                            block.resize(count);
                            std::copy_n(chunk.begin(), count, block.begin());
                            if (queue->chunks.size() >= queue->maximum_chunks) {
                                auto evicted = std::move(queue->chunks.front());
                                queue->chunks.pop_front();
                                if (queue->free_chunks.size() < queue->maximum_chunks + 4) {
                                    evicted.clear();
                                    queue->free_chunks.push_back(std::move(evicted));
                                }
                                ++queue->discontinuities;
                                ++counters.rx_drops;
                            }
                            queue->chunks.push_back(std::move(block));
                        }
                        queue->ready.notify_one();
                    }
                } catch (const std::exception& error) {
                    ++counters.errors;
                    publish_status(telemetry, counters, false, false, error.what());
                    running = false;
                }
                {
                    std::lock_guard<std::mutex> lock(queue->mutex);
                    queue->closed = true;
                }
                queue->ready.notify_all();
            });

            decoder = std::thread([&, queue] {
                try {
                    ofdm::RxPipeline pipeline;
                    pipeline.set_mmse(true);
                    uint64_t discontinuities = 0;
                    while (true) {
                        std::vector<cf32> block;
                        uint64_t current_discontinuities = 0;
                        {
                            std::unique_lock<std::mutex> lock(queue->mutex);
                            queue->ready.wait(lock, [&] { return !queue->chunks.empty() || queue->closed || !running; });
                            if (queue->chunks.empty()) {
                                if (queue->closed || !running) break;
                                continue;
                            }
                            block = std::move(queue->chunks.front());
                            queue->chunks.pop_front();
                            current_discontinuities = queue->discontinuities;
                        }
                        if (current_discontinuities != discontinuities) {
                            // A dropped capture block creates an unknown gap;
                            // discard the partial frame and reacquire on the
                            // next contiguous block.
                            pipeline.reset();
                            discontinuities = current_discontinuities;
                        }
                        const auto results = pipeline.process(block.data(), static_cast<int>(block.size()));
                        const auto diagnostics = pipeline.diagnostics();
                        counters.rx_stf.store(diagnostics.stf_candidates);
                        counters.rx_ltf.store(diagnostics.ltf_locks);
                        counters.rx_csi_rejects.store(diagnostics.csi_rejects);
                        counters.rx_control_failures.store(diagnostics.control_failures);
                        counters.rx_header_failures.store(diagnostics.header_failures);
                        counters.rx_last_csi_db.store(diagnostics.last_csi_quality_db);
                        counters.rx_max_csi_db.store(diagnostics.max_csi_quality_db);
                        counters.rx_last_cfo_hz.store(diagnostics.last_cfo_hz);
                        for (const auto& result : results) {
                            publish_result(result, output.get(), telemetry, counters);
                        }
                        block.clear();
                        {
                            std::lock_guard<std::mutex> lock(queue->mutex);
                            if (queue->free_chunks.size() < queue->maximum_chunks + 4) {
                                queue->free_chunks.push_back(std::move(block));
                            }
                        }
                    }
                } catch (const std::exception& error) {
                    ++counters.errors;
                    publish_status(telemetry, counters, false, false, error.what());
                    running = false;
                }
            });
        }

        // TX owns its ZeroMQ socket, frame builder, and blocking SDR writes in
        // one dedicated worker.  The main thread remains responsive for
        // status/UI telemetry exactly as the RX acquisition and decoder
        // workers do.  TX stays enabled until STOP/process shutdown; there is
        // no time-based authorization expiry.
        const bool tx_enabled = options.simulate || options.role != Options::Role::Rx;
        if (tx_enabled) {
            transmitter = std::thread([&] {
                try {
                    ofdm::ZmqTxInterface input(options.tx_endpoint);
                    ofdm::TxPipeline tx_pipeline;
                    ofdm::RxPipeline simulation_rx;
                    simulation_rx.set_mmse(true);
                    uint32_t next_id = 1;
                    auto next_demo = std::chrono::steady_clock::now();
                    float demo_phase = 0.0f;
                    while (running) {
                        const auto now = std::chrono::steady_clock::now();
                        std::string metadata;
                        std::vector<cf32> data;
                        const bool network_input = input.receive(metadata, data);
                        if (!network_input && options.demo_source && now >= next_demo) {
                            constexpr float pi = 3.14159265358979323846f;
                            data.resize(1024);
                            for (std::size_t index = 0; index < data.size(); ++index) {
                                const float position = static_cast<float>(index % ofdm::NUM_DATA_SC) /
                                                       static_cast<float>(ofdm::NUM_DATA_SC);
                                const float amplitude = 0.32f + 0.12f *
                                    std::sin(2.0f * pi * (position * 3.0f + demo_phase * 0.13f));
                                const float angle = 2.0f * pi * position + demo_phase;
                                data[index] = std::polar(amplitude, angle);
                            }
                            metadata = "{\"version\":1,\"burst_id\":" + std::to_string(next_id) +
                                       ",\"source\":\"demo\"}";
                            demo_phase = std::fmod(demo_phase + 0.17f, 2.0f * pi);
                            next_demo = now + std::chrono::milliseconds(250);
                        } else if (!network_input) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(2));
                            continue;
                        }

                        const auto burst_id = burst_id_from_json(metadata, next_id++);
                        auto burst = tx_pipeline.build_burst(data.data(), data.size(), burst_id);
                        if (radio) radio->write_burst(burst.samples.data(), burst.samples.size());
                        else if (tx_radio) tx_radio->write_burst(burst.samples.data(), burst.samples.size());
                        telemetry.iq(ofdm::TelemetryKind::TxIq, burst.samples.data(), burst.samples.size());
                        ++counters.tx_bursts;
                        counters.tx_samples.fetch_add(burst.samples.size());

                        if (options.simulate) {
                            telemetry.iq(ofdm::TelemetryKind::RxIq, burst.samples.data(), burst.samples.size());
                            counters.rx_samples.fetch_add(burst.samples.size());
                            const auto results = simulation_rx.process(
                                burst.samples.data(), static_cast<int>(burst.samples.size()));
                            for (const auto& result : results) {
                                publish_result(result, output.get(), telemetry, counters);
                            }
                        }
                    }
                } catch (const std::exception& error) {
                    ++counters.errors;
                    publish_status(telemetry, counters, options.simulate, false, error.what());
                    running = false;
                }
            });
        }

        auto next_status = std::chrono::steady_clock::now();
        while (running) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= next_status) {
                publish_status(telemetry, counters, options.simulate, tx_enabled);
                next_status = now + std::chrono::seconds(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (transmitter.joinable()) transmitter.join();
        if (receiver.joinable()) receiver.join();
        if (rx_queue) rx_queue->ready.notify_all();
        if (decoder.joinable()) decoder.join();
        publish_status(telemetry, counters, options.simulate, false);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ofdm-workbench-service: " << error.what() << '\n';
        return 1;
    }
}
