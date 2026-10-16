#pragma once

#include "dsp/ofdm_core.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ofdm::wifi_anchor {

// The anchor is a standards-compatible 802.11a/g legacy 6 Mbps PHY frame.
// Its PSDU is a broadcast data MPDU with a normal MAC header and LLC/SNAP;
// the versioned project control record lives in the MAC payload and is
// followed by a normal FCS. BPSK 1/2 is deliberately used so this control
// information is substantially more robust than the opaque complex payload.
constexpr std::size_t kMetadataBytes = 20;
constexpr std::size_t kMacHeaderBytes = 24;
constexpr std::size_t kLlcSnapBytes = 8;
constexpr std::size_t kControlOffset = kMacHeaderBytes + kLlcSnapBytes;
constexpr std::size_t kPsduBytes = kControlOffset + kMetadataBytes + 4;
constexpr std::size_t kDataBitsPerSymbol = 24; // 48 carriers * BPSK * 1/2
constexpr std::size_t kDataSymbols =
    (16 + kPsduBytes * 8 + 6 + kDataBitsPerSymbol - 1) / kDataBitsPerSymbol;
constexpr std::size_t kFrameSamples = 320 + 80 + kDataSymbols * SYM_SAMP;
constexpr std::size_t kGuardSamples = 32;

constexpr std::uint8_t kExtensionVersion = 2;
constexpr std::uint8_t kCf32Format = 1;
constexpr std::uint8_t kPilotSchemeLegacy = 1;

struct ExtensionControl {
    std::uint32_t burst_id{};
    std::uint32_t original_length{};
    std::uint16_t num_data_symbols{};
};

std::array<std::uint8_t, kMetadataBytes> encode_extension_control(
    const ExtensionControl& control);
bool decode_extension_control(
    const std::array<std::uint8_t, kMetadataBytes>& metadata,
    ExtensionControl& control);

struct DecodeResult {
    bool ok{false};
    std::array<std::uint8_t, kMetadataBytes> metadata{};
    std::size_t frame_samples{};
    float cfo_hz{};
    float cfo_rad{};
    float noise_var{};
    cf32 csi[FFT_SIZE]{};
};

// Build a normal legacy Wi-Fi 6 Mbps frame.  The metadata bytes are carried
// in the PSDU; the caller may append a project-specific waveform afterwards.
std::vector<cf32> build(const std::array<std::uint8_t, kMetadataBytes>& metadata);

// Decode a legacy anchor at an already detected STF start.  This function is
// deliberately independent from the raw-I/Q decoder so a receiver can first
// use the Marconi-style legacy synchronization path and only then parse the
// project payload.
DecodeResult decode(const cf32* samples, std::size_t count, std::size_t start = 0,
                    std::size_t maximum_psdu = 64);

} // namespace ofdm::wifi_anchor
