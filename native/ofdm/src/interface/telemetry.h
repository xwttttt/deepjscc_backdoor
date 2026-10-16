#pragma once

#include "dsp/ofdm_core.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

namespace ofdm {

enum class TelemetryKind : uint16_t {
    Status = 1,
    TxIq = 2,
    RxIq = 3,
    Csi = 4,
};

// Thread-safe binary stream written to stdout and consumed by the Tauri host.
// Header: magic "OFDM", version u16, kind u16, sequence u64, payload_size u32.
class TelemetryWriter {
public:
    void json(TelemetryKind kind, const std::string& value);
    void iq(TelemetryKind kind, const cf32* samples, std::size_t count,
            std::size_t maximum_points = 512);

private:
    void frame(TelemetryKind kind, const void* payload, std::size_t size);
    std::mutex mutex_;
    uint64_t sequence_{};
};

} // namespace ofdm
