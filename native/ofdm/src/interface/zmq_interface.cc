#include "interface/zmq_interface.h"
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <cerrno>

namespace ofdm {

// --- TX Interface (PULL socket) ---
ZmqTxInterface::ZmqTxInterface(const std::string& endpoint) {
    ctx_ = zmq_ctx_new();
    socket_ = zmq_socket(ctx_, ZMQ_PULL);
    int hwm = 4;
    zmq_setsockopt(socket_, ZMQ_RCVHWM, &hwm, sizeof(hwm));
    int64_t maxmsg = 8388608 * sizeof(cf32) + 4096;  // max message size
    zmq_setsockopt(socket_, ZMQ_MAXMSGSIZE, &maxmsg, sizeof(maxmsg));
    if (zmq_bind(socket_, endpoint.c_str()) != 0) {
        throw std::runtime_error("ZmqTxInterface: bind failed: " + endpoint);
    }
}

ZmqTxInterface::~ZmqTxInterface() {
    zmq_close(socket_);
    zmq_ctx_destroy(ctx_);
}

bool ZmqTxInterface::receive(std::string& json_out, std::vector<cf32>& data_out) {
    zmq_msg_t msg;

    // Part 1: JSON metadata
    zmq_msg_init(&msg);
    if (zmq_msg_recv(&msg, socket_, ZMQ_DONTWAIT) < 0) {
        zmq_msg_close(&msg);
        return false;
    }
    json_out.assign((const char*)zmq_msg_data(&msg), zmq_msg_size(&msg));
    const int more = zmq_msg_more(&msg);
    zmq_msg_close(&msg);
    if (!more) return false;

    // Part 2: CF32 data (little-endian float pairs)
    zmq_msg_init(&msg);
    if (zmq_msg_recv(&msg, socket_, 0) < 0) {
        zmq_msg_close(&msg);
        return false;
    }
    size_t nbytes = zmq_msg_size(&msg);
    if (nbytes == 0 || nbytes % sizeof(cf32) != 0 || nbytes > 8388608ULL * sizeof(cf32)) {
        zmq_msg_close(&msg);
        return false;
    }
    size_t num_cf32 = nbytes / sizeof(cf32);
    data_out.resize(num_cf32);
    std::memcpy(data_out.data(), zmq_msg_data(&msg), nbytes);
    bool valid = true;
    for (size_t i = 0; i < num_cf32; i++) {
        if (!std::isfinite(data_out[i].real()) || !std::isfinite(data_out[i].imag())) {
            valid = false;
            break;
        }
    }
    zmq_msg_close(&msg);

    if (!valid) data_out.clear();
    return valid && num_cf32 > 0;
}

// --- RX Interface (PUSH socket) ---
ZmqRxInterface::ZmqRxInterface(const std::string& endpoint) {
    ctx_ = zmq_ctx_new();
    socket_ = zmq_socket(ctx_, ZMQ_PUSH);
    int hwm = 4;
    zmq_setsockopt(socket_, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    if (zmq_bind(socket_, endpoint.c_str()) != 0) {
        throw std::runtime_error("ZmqRxInterface: bind failed: " + endpoint);
    }
}

ZmqRxInterface::~ZmqRxInterface() {
    zmq_close(socket_);
    zmq_ctx_destroy(ctx_);
}

bool ZmqRxInterface::send(const std::string& json_meta, const cf32* csi,
                         const cf32* data, size_t data_len) {
    // Part 1: JSON metadata
    zmq_msg_t msg;
    zmq_msg_init_size(&msg, json_meta.size());
    memcpy(zmq_msg_data(&msg), json_meta.data(), json_meta.size());
    const int first = zmq_msg_send(&msg, socket_, ZMQ_SNDMORE | ZMQ_DONTWAIT);
    zmq_msg_close(&msg);
    if (first < 0) return false;

    // Part 2: CSI (64 CF32)
    zmq_msg_init_size(&msg, FFT_SIZE * sizeof(cf32));
    memcpy(zmq_msg_data(&msg), csi, FFT_SIZE * sizeof(cf32));
    if (zmq_msg_send(&msg, socket_, ZMQ_SNDMORE) < 0) {
        zmq_msg_close(&msg);
        return false;
    }
    zmq_msg_close(&msg);

    // Part 3: Recovered data
    zmq_msg_init_size(&msg, data_len * sizeof(cf32));
    memcpy(zmq_msg_data(&msg), data, data_len * sizeof(cf32));
    const bool sent = zmq_msg_send(&msg, socket_, 0) >= 0;
    zmq_msg_close(&msg);
    return sent;
}

void ZmqRxInterface::send_telemetry(const std::string& json) {
    zmq_msg_t msg;
    zmq_msg_init_size(&msg, json.size());
    memcpy(zmq_msg_data(&msg), json.data(), json.size());
    zmq_msg_send(&msg, socket_, 0);
    zmq_msg_close(&msg);
}

} // namespace ofdm
