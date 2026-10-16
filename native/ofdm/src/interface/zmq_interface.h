#pragma once
#include "dsp/ofdm_core.h"
#include <zmq.h>
#include <string>
#include <vector>
#include <cstdint>

namespace ofdm {

// ZeroMQ interface for TX and RX services
// TX: PULL socket, receives multipart (JSON + CF32 data)
// RX: PUSH socket, sends multipart (JSON metadata + CSI + recovered data)

class ZmqTxInterface {
public:
    ZmqTxInterface(const std::string& endpoint = "tcp://127.0.0.1:5555");
    ~ZmqTxInterface();

    // Try to receive a burst request. Returns true if received.
    // json_out: UTF-8 JSON metadata
    // data_out: CF32 samples
    bool receive(std::string& json_out, std::vector<cf32>& data_out);

private:
    void* ctx_;
    void* socket_;
};

class ZmqRxInterface {
public:
    ZmqRxInterface(const std::string& endpoint = "tcp://127.0.0.1:5556");
    ~ZmqRxInterface();

    // Send a burst result as 3-part message
    // json_meta: JSON metadata
    // csi: 64 CF32 channel state info
    // data: recovered CF32 data
    bool send(const std::string& json_meta, const cf32* csi, const cf32* data, size_t data_len);

    // Send error/status telemetry
    void send_telemetry(const std::string& json);

private:
    void* ctx_;
    void* socket_;
};

} // namespace ofdm
