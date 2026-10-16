#include "interface/telemetry.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ofdm {
namespace {

void put_u16(unsigned char* out, uint16_t value) {
    out[0] = static_cast<unsigned char>(value);
    out[1] = static_cast<unsigned char>(value >> 8);
}
void put_u32(unsigned char* out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out[i] = static_cast<unsigned char>(value >> (8 * i));
}
void put_u64(unsigned char* out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out[i] = static_cast<unsigned char>(value >> (8 * i));
}

} // namespace

void TelemetryWriter::frame(TelemetryKind kind, const void* payload, std::size_t size) {
    if (size > UINT32_MAX) throw std::length_error("telemetry payload too large");
    std::lock_guard<std::mutex> lock(mutex_);
    std::array<unsigned char, 20> header{'O', 'F', 'D', 'M'};
    put_u16(header.data() + 4, 1);
    put_u16(header.data() + 6, static_cast<uint16_t>(kind));
    put_u64(header.data() + 8, ++sequence_);
    put_u32(header.data() + 16, static_cast<uint32_t>(size));
    if (std::fwrite(header.data(), 1, header.size(), stdout) != header.size() ||
        (size && std::fwrite(payload, 1, size, stdout) != size)) {
        throw std::runtime_error("telemetry stdout write failed");
    }
    std::fflush(stdout);
}

void TelemetryWriter::json(TelemetryKind kind, const std::string& value) {
    frame(kind, value.data(), value.size());
}

void TelemetryWriter::iq(TelemetryKind kind, const cf32* samples, std::size_t count,
                         std::size_t maximum_points) {
    if (!samples || count == 0 || maximum_points == 0) return;
    const auto points = std::min(count, maximum_points);
    const auto stride = std::max<std::size_t>(1, count / points);
    std::vector<cf32> output;
    output.reserve(points);
    for (std::size_t i = 0; i < count && output.size() < points; i += stride) output.push_back(samples[i]);
    frame(kind, output.data(), output.size() * sizeof(cf32));
}

} // namespace ofdm
