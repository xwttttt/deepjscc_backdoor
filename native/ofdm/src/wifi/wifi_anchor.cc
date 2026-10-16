#include "wifi/wifi_anchor.h"

#include <liquid/liquid.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ofdm::wifi_anchor {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInvSqrt2 = 0.7071067811865475f;
constexpr std::array<int, 48> kDataSc{{
    -26, -25, -24, -23, -22, -20, -19, -18, -17, -16, -15, -14,
    -13, -12, -11, -10, -9, -8, -6, -5, -4, -3, -2, -1,
      1,   2,   3,   4,   5,   6,   8,   9,  10,  11,  12,  13,
     14,  15,  16,  17,  18,  19,  20,  22,  23,  24,  25,  26
}};
constexpr std::array<int, 4> kPilotSc{{-21, -7, 7, 21}};
constexpr std::array<float, 4> kPilotBase{{1.0f, 1.0f, 1.0f, -1.0f}};
using Bits = std::vector<std::uint8_t>;

std::uint8_t parity(std::uint32_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::uint8_t>(__builtin_parity(value));
#else
    value ^= value >> 16; value ^= value >> 8; value ^= value >> 4;
    value &= 0xf;
    return static_cast<std::uint8_t>((0x6996u >> value) & 1u);
#endif
}

void fft64(const cf32* input, cf32* output, int direction) {
    auto plan = fft_create_plan(
        FFT_SIZE,
        const_cast<liquid_float_complex*>(reinterpret_cast<const liquid_float_complex*>(input)),
        reinterpret_cast<liquid_float_complex*>(output), direction, 0);
    if (!plan) throw std::runtime_error("Liquid-DSP could not create Wi-Fi FFT plan");
    fft_execute(plan);
    fft_destroy_plan(plan);
}

void append_symbol(std::vector<cf32>& output, const cf32* frequency) {
    cf32 time[FFT_SIZE];
    fft64(frequency, time, LIQUID_FFT_BACKWARD);
    // Marconi's PHY uses an unnormalised inverse FFT and normalises by the
    // number of active legacy tones (52 = 48 data + 4 pilots).
    const float scale = 1.0f / std::sqrt(52.0f);
    for (auto& value : time) value *= scale;
    output.insert(output.end(), time + FFT_SIZE - CP_LEN, time + FFT_SIZE);
    output.insert(output.end(), time, time + FFT_SIZE);
}

std::array<cf32, FFT_SIZE> short_frequency() {
    std::array<cf32, FFT_SIZE> result{};
    const cf32 positive{kInvSqrt2, kInvSqrt2};
    const cf32 negative = -positive;
    result[sc_to_fft(-24)] = positive;
    result[sc_to_fft(-20)] = negative;
    result[sc_to_fft(-16)] = positive;
    result[sc_to_fft(-12)] = negative;
    result[sc_to_fft(-8)] = negative;
    result[sc_to_fft(-4)] = positive;
    result[sc_to_fft(4)] = negative;
    result[sc_to_fft(8)] = negative;
    result[sc_to_fft(12)] = positive;
    result[sc_to_fft(16)] = positive;
    result[sc_to_fft(20)] = positive;
    result[sc_to_fft(24)] = positive;
    return result;
}

std::array<cf32, FFT_SIZE> long_frequency() {
    constexpr std::array<int, 53> sequence{{
        1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
        1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,
        1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1,
        -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1
    }};
    std::array<cf32, FFT_SIZE> result{};
    for (int sc = -26; sc <= 26; ++sc) result[sc_to_fft(sc)] = static_cast<float>(sequence[sc + 26]);
    return result;
}

std::vector<cf32> legacy_preamble() {
    const auto short_freq = short_frequency();
    const auto long_freq = long_frequency();
    cf32 short_time[FFT_SIZE], long_time[FFT_SIZE];
    fft64(short_freq.data(), short_time, LIQUID_FFT_BACKWARD);
    fft64(long_freq.data(), long_time, LIQUID_FFT_BACKWARD);
    for (auto& value : short_time) value *= 1.0f / std::sqrt(12.0f);
    for (auto& value : long_time) value *= 1.0f / std::sqrt(52.0f);
    std::vector<cf32> result;
    result.reserve(320);
    for (int repeat = 0; repeat < 10; ++repeat) result.insert(result.end(), short_time, short_time + 16);
    result.insert(result.end(), long_time + 32, long_time + 64);
    result.insert(result.end(), long_time, long_time + 64);
    result.insert(result.end(), long_time, long_time + 64);
    return result;
}

Bits bytes_to_bits(const std::vector<std::uint8_t>& bytes) {
    Bits result;
    result.reserve(bytes.size() * 8);
    for (const auto value : bytes) for (unsigned bit = 0; bit < 8; ++bit) result.push_back((value >> bit) & 1u);
    return result;
}

std::vector<std::uint8_t> bits_to_bytes(const Bits& bits) {
    std::vector<std::uint8_t> result((bits.size() + 7) / 8);
    for (std::size_t i = 0; i < bits.size(); ++i) result[i / 8] |= static_cast<std::uint8_t>((bits[i] & 1u) << (i % 8));
    return result;
}

std::uint32_t crc32(const std::uint8_t* data, std::size_t size) {
    std::uint32_t value = 0xffffffffu;
    for (std::size_t i = 0; i < size; ++i) {
        value ^= data[i];
        for (unsigned bit = 0; bit < 8; ++bit) value = (value >> 1) ^ (0xedb88320u & (0u - (value & 1u)));
    }
    return value ^ 0xffffffffu;
}

Bits convolutional_encode(const Bits& input) {
    Bits result;
    result.reserve(input.size() * 2);
    std::uint32_t state = 0;
    for (const auto bit : input) {
        const auto shift = ((state << 1) | (bit & 1u)) & 0x7fu;
        result.push_back(parity(shift & 0x6du));
        result.push_back(parity(shift & 0x4fu));
        state = shift & 0x3fu;
    }
    return result;
}

Bits viterbi_decode(const std::vector<std::int8_t>& encoded) {
    if (encoded.size() % 2) return {};
    constexpr float infinity = 1.0e20f;
    const std::size_t steps = encoded.size() / 2;
    std::array<float, 64> metrics{}, next{};
    metrics.fill(infinity); metrics[0] = 0.0f;
    std::vector<std::uint8_t> previous_state(steps * 64), previous_bit(steps * 64);
    for (std::size_t step = 0; step < steps; ++step) {
        next.fill(infinity);
        for (std::uint32_t state = 0; state < 64; ++state) {
            if (metrics[state] >= infinity * 0.5f) continue;
            for (std::uint32_t bit = 0; bit < 2; ++bit) {
                const auto shift = ((state << 1) | bit) & 0x7fu;
                const auto next_state = shift & 0x3fu;
                float branch = 0.0f;
                if (encoded[2 * step] >= 0) branch += encoded[2 * step] != parity(shift & 0x6du);
                if (encoded[2 * step + 1] >= 0) branch += encoded[2 * step + 1] != parity(shift & 0x4fu);
                if (metrics[state] + branch < next[next_state]) {
                    next[next_state] = metrics[state] + branch;
                    previous_state[step * 64 + next_state] = static_cast<std::uint8_t>(state);
                    previous_bit[step * 64 + next_state] = static_cast<std::uint8_t>(bit);
                }
            }
        }
        metrics = next;
    }
    if (metrics[0] >= infinity * 0.5f) return {};
    Bits result(steps);
    std::uint32_t state = 0;
    for (std::size_t step = steps; step-- > 0;) {
        result[step] = previous_bit[step * 64 + state];
        state = previous_state[step * 64 + state];
    }
    return result;
}

Bits interleave(const Bits& input, unsigned bits_per_carrier) {
    constexpr std::size_t columns = 16;
    const std::size_t ncbps = input.size();
    const std::size_t rows = ncbps / columns;
    const std::size_t s = std::max<std::size_t>(bits_per_carrier / 2, 1);
    Bits result(ncbps);
    for (std::size_t k = 0; k < ncbps; ++k) {
        const std::size_t i = rows * (k % columns) + k / columns;
        const std::size_t j = s * (i / s) + (i + ncbps - (columns * i) / ncbps) % s;
        result[j] = input[k];
    }
    return result;
}

Bits deinterleave(const Bits& input, unsigned bits_per_carrier) {
    constexpr std::size_t columns = 16;
    const std::size_t ncbps = input.size();
    const std::size_t rows = ncbps / columns;
    const std::size_t s = std::max<std::size_t>(bits_per_carrier / 2, 1);
    Bits result(ncbps);
    for (std::size_t k = 0; k < ncbps; ++k) {
        const std::size_t i = rows * (k % columns) + k / columns;
        const std::size_t j = s * (i / s) + (i + ncbps - (columns * i) / ncbps) % s;
        result[k] = input[j];
    }
    return result;
}

Bits scramble(const Bits& input) {
    Bits result(input.size());
    std::uint8_t state = 0x5d;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto feedback = static_cast<std::uint8_t>(((state >> 6) ^ (state >> 3)) & 1u);
        result[i] = static_cast<std::uint8_t>(input[i] ^ feedback);
        state = static_cast<std::uint8_t>(((state << 1) & 0x7e) | feedback);
    }
    return result;
}

cf32 qam_symbol(const std::uint8_t* bits, unsigned count) {
    const auto sign = [](std::uint8_t bit) { return bit ? 1.0f : -1.0f; };
    if (count == 1) return {sign(bits[0]), 0.0f};
    if (count != 6) throw std::invalid_argument("Wi-Fi anchor supports BPSK and 64-QAM only");
    const float real = sign(bits[0]) * (4.0f - sign(bits[1]) * (2.0f - sign(bits[2])));
    const float imag = sign(bits[3]) * (4.0f - sign(bits[4]) * (2.0f - sign(bits[5])));
    return {real / std::sqrt(42.0f), imag / std::sqrt(42.0f)};
}

Bits demap_qam(cf32 value, unsigned count) {
    Bits best(count);
    float best_distance = std::numeric_limits<float>::max();
    const unsigned combinations = 1u << count;
    for (unsigned mask = 0; mask < combinations; ++mask) {
        Bits bits(count);
        for (unsigned bit = 0; bit < count; ++bit) bits[bit] = static_cast<std::uint8_t>((mask >> bit) & 1u);
        const float distance = std::norm(value - qam_symbol(bits.data(), count));
        if (distance < best_distance) { best_distance = distance; best = bits; }
    }
    return best;
}

int pilot_polarity(std::size_t symbol_index) {
    std::uint8_t state = 0x7f, feedback = 0;
    for (std::size_t i = 0; i <= symbol_index % 127; ++i) {
        feedback = static_cast<std::uint8_t>(((state >> 6) ^ (state >> 3)) & 1u);
        state = static_cast<std::uint8_t>(((state << 1) & 0x7e) | feedback);
    }
    return feedback ? -1 : 1;
}

std::array<cf32, FFT_SIZE> equalize(
    const cf32* time, const cf32* csi, float cfo_rad, std::size_t symbol_index, float noise_var) {
    cf32 corrected[FFT_SIZE], frequency[FFT_SIZE];
    for (int i = 0; i < FFT_SIZE; ++i) {
        const float phase = static_cast<float>(i) * cfo_rad;
        corrected[i] = time[i] * cf32(std::cos(phase), std::sin(phase));
    }
    fft64(corrected, frequency, LIQUID_FFT_FORWARD);
    std::array<cf32, FFT_SIZE> result{};
    for (int sc = -26; sc <= 26; ++sc) {
        if (sc == 0) continue;
        const int bin = sc_to_fft(sc);
        const float h2 = std::norm(csi[bin]);
        if (h2 > 1.0e-12f) result[bin] = frequency[bin] * std::conj(csi[bin]) / (h2 + std::max(noise_var, 0.0f));
    }
    const float polarity = static_cast<float>(pilot_polarity(symbol_index));
    cf32 pilot_sum{};
    for (std::size_t i = 0; i < kPilotSc.size(); ++i) pilot_sum += result[sc_to_fft(kPilotSc[i])] * (kPilotBase[i] * polarity);
    const auto correction = std::polar(1.0f, -std::arg(pilot_sum));
    for (auto& value : result) value *= correction;
    return result;
}

float coarse_cfo(const cf32* samples, std::size_t ltf_start) {
    cf32 sum{};
    for (int i = 0; i < 64; ++i) sum += samples[ltf_start - 80 + i] * std::conj(samples[ltf_start - 80 + i + 16]);
    return std::arg(sum) / 16.0f;
}

float fine_cfo(const cf32* samples, std::size_t ltf_start, float coarse) {
    cf32 sum{};
    for (int i = 0; i < 64; ++i) {
        const auto a = ltf_start + 32 + static_cast<std::size_t>(i);
        const auto b = ltf_start + 96 + static_cast<std::size_t>(i);
        const auto ca = samples[a] * std::polar(1.0f, static_cast<float>(a) * coarse);
        const auto cb = samples[b] * std::polar(1.0f, static_cast<float>(b) * coarse);
        sum += ca * std::conj(cb);
    }
    return coarse + std::arg(sum) / 64.0f;
}

void estimate_channel(const cf32* samples, std::size_t ltf_start, float cfo, cf32* csi, float& noise_var) {
    cf32 a[64], b[64], af[64], bf[64];
    for (int i = 0; i < 64; ++i) {
        const auto ia = ltf_start + 32 + static_cast<std::size_t>(i);
        const auto ib = ltf_start + 96 + static_cast<std::size_t>(i);
        a[i] = samples[ia] * std::polar(1.0f, static_cast<float>(ia) * cfo);
        b[i] = samples[ib] * std::polar(1.0f, static_cast<float>(ib) * cfo);
    }
    fft64(a, af, LIQUID_FFT_FORWARD); fft64(b, bf, LIQUID_FFT_FORWARD);
    const auto reference = long_frequency();
    double error = 0.0; int count = 0;
    for (int bin = 0; bin < FFT_SIZE; ++bin) {
        if (std::norm(reference[bin]) > 0.0f) {
            csi[bin] = (af[bin] + bf[bin]) * 0.5f / reference[bin];
            error += std::norm(af[bin] - bf[bin]) * 0.5;
            ++count;
        } else csi[bin] = cf32{};
    }
    noise_var = static_cast<float>(std::max(error / std::max(1, count), 1.0e-12));
}

} // namespace

std::array<std::uint8_t, kMetadataBytes> encode_extension_control(
    const ExtensionControl& control) {
    std::array<std::uint8_t, kMetadataBytes> result{};
    result[0] = 'O'; result[1] = 'F'; result[2] = 'D'; result[3] = 'M';
    result[4] = kExtensionVersion;
    result[5] = kCf32Format;
    result[6] = static_cast<std::uint8_t>(FFT_SIZE);
    result[7] = static_cast<std::uint8_t>(CP_LEN);
    for (unsigned byte = 0; byte < 4; ++byte) {
        result[8 + byte] = static_cast<std::uint8_t>(control.burst_id >> (8 * byte));
        result[12 + byte] = static_cast<std::uint8_t>(control.original_length >> (8 * byte));
    }
    result[16] = static_cast<std::uint8_t>(control.num_data_symbols);
    result[17] = static_cast<std::uint8_t>(control.num_data_symbols >> 8);
    result[18] = static_cast<std::uint8_t>(NUM_DATA_SC);
    result[19] = kPilotSchemeLegacy;
    return result;
}

bool decode_extension_control(
    const std::array<std::uint8_t, kMetadataBytes>& metadata,
    ExtensionControl& control) {
    if (metadata[0] != 'O' || metadata[1] != 'F' ||
        metadata[2] != 'D' || metadata[3] != 'M' ||
        metadata[4] != kExtensionVersion || metadata[5] != kCf32Format ||
        metadata[6] != FFT_SIZE || metadata[7] != CP_LEN ||
        metadata[18] != NUM_DATA_SC || metadata[19] != kPilotSchemeLegacy) {
        return false;
    }
    control = {};
    for (unsigned byte = 0; byte < 4; ++byte) {
        control.burst_id |= static_cast<std::uint32_t>(metadata[8 + byte]) << (8 * byte);
        control.original_length |= static_cast<std::uint32_t>(metadata[12 + byte]) << (8 * byte);
    }
    control.num_data_symbols = static_cast<std::uint16_t>(metadata[16]) |
        (static_cast<std::uint16_t>(metadata[17]) << 8);
    const auto expected = (control.original_length + NUM_DATA_SC - 1U) / NUM_DATA_SC;
    return control.original_length > 0 && control.num_data_symbols > 0 &&
        control.num_data_symbols == expected;
}

std::vector<cf32> build(const std::array<std::uint8_t, kMetadataBytes>& metadata) {
    // IEEE 802.11 data MPDU: data frame control, duration, broadcast DA,
    // locally administered SA/BSSID, sequence control, then LLC/SNAP with the
    // locally experimental EtherType 0x88b5.
    std::vector<std::uint8_t> psdu{
        0x08, 0x00, 0x00, 0x00,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x02, 0x4f, 0x46, 0x44, 0x4d, 0x01,
        0x02, 0x4f, 0x46, 0x44, 0x4d, 0x01,
        0x00, 0x00,
        0xaa, 0xaa, 0x03, 0x00, 0x00, 0x00, 0x88, 0xb5,
    };
    psdu.insert(psdu.end(), metadata.begin(), metadata.end());
    const auto burst_id = static_cast<std::uint16_t>(metadata[8]) |
        (static_cast<std::uint16_t>(metadata[9]) << 8);
    const auto sequence_control = static_cast<std::uint16_t>((burst_id & 0x0fffu) << 4);
    psdu[22] = static_cast<std::uint8_t>(sequence_control);
    psdu[23] = static_cast<std::uint8_t>(sequence_control >> 8);
    const auto fcs = crc32(psdu.data(), psdu.size());
    for (unsigned shift = 0; shift < 32; shift += 8) psdu.push_back(static_cast<std::uint8_t>(fcs >> shift));

    auto result = legacy_preamble();

    // Legacy SIGNAL: 6 Mbps (1101), BPSK 1/2, terminated.
    Bits signal(24, 0);
    signal[0] = 1; signal[1] = 1; signal[2] = 0; signal[3] = 1;
    for (unsigned bit = 0; bit < 12; ++bit) signal[5 + bit] = static_cast<std::uint8_t>((kPsduBytes >> bit) & 1u);
    for (unsigned bit = 0; bit < 17; ++bit) signal[17] ^= signal[bit];
    const auto signal_coded = interleave(convolutional_encode(signal), 1);
    std::array<cf32, FFT_SIZE> signal_frequency{};
    for (std::size_t i = 0; i < kDataSc.size(); ++i) signal_frequency[sc_to_fft(kDataSc[i])] = qam_symbol(&signal_coded[i], 1);
    for (std::size_t i = 0; i < kPilotSc.size(); ++i) signal_frequency[sc_to_fft(kPilotSc[i])] = kPilotBase[i];
    append_symbol(result, signal_frequency.data());

    Bits input(kDataSymbols * kDataBitsPerSymbol, 0);
    const auto payload_bits = bytes_to_bits(psdu);
    std::copy(payload_bits.begin(), payload_bits.end(), input.begin() + 16);
    auto scrambled = scramble(input);
    const auto tail_start = 16 + payload_bits.size();
    std::fill(scrambled.begin() + static_cast<std::ptrdiff_t>(tail_start),
              scrambled.begin() + static_cast<std::ptrdiff_t>(tail_start + 6), 0);
    const auto coded = convolutional_encode(scrambled);
    for (std::size_t symbol = 0; symbol < kDataSymbols; ++symbol) {
        const Bits block(coded.begin() + static_cast<std::ptrdiff_t>(symbol * 48),
                         coded.begin() + static_cast<std::ptrdiff_t>((symbol + 1) * 48));
        const auto interleaved = interleave(block, 1);
        std::array<cf32, FFT_SIZE> frequency{};
        for (std::size_t i = 0; i < kDataSc.size(); ++i) frequency[sc_to_fft(kDataSc[i])] = qam_symbol(&interleaved[i], 1);
        const float polarity = static_cast<float>(pilot_polarity(symbol + 1));
        for (std::size_t i = 0; i < kPilotSc.size(); ++i) frequency[sc_to_fft(kPilotSc[i])] = kPilotBase[i] * polarity;
        append_symbol(result, frequency.data());
    }

    // Leave the anchor at the same fixed IFFT convention as the following
    // project block.  TxPipeline applies one common peak limit after it has
    // appended the complete hybrid burst.  Per-anchor normalization here is
    // tempting, but it makes the L-LTF gain depend on the metadata bits and
    // then the later raw-I/Q block no longer shares the measured CSI scale.
    return result;
}

DecodeResult decode(const cf32* samples, std::size_t count, std::size_t start, std::size_t maximum_psdu) {
    DecodeResult result;
    if (!samples || maximum_psdu < kPsduBytes || start + kFrameSamples > count) return result;
    const auto ltf_start = start + 160;
    const float coarse = coarse_cfo(samples, ltf_start);
    const float cfo = fine_cfo(samples, ltf_start, coarse);
    result.cfo_rad = cfo;
    result.cfo_hz = cfo * static_cast<float>(SAMPLE_RATE) / (2.0f * kPi);
    estimate_channel(samples, ltf_start, cfo, result.csi, result.noise_var);

    const auto signal = equalize(samples + start + 320 + CP_LEN, result.csi, cfo, 0, result.noise_var);
    Bits signal_interleaved;
    for (const auto sc : kDataSc) signal_interleaved.push_back(signal[sc_to_fft(sc)].real() >= 0.0f ? 1u : 0u);
    const auto signal_coded = deinterleave(signal_interleaved, 1);
    std::vector<std::int8_t> signal_soft(signal_coded.begin(), signal_coded.end());
    const auto signal_bits = viterbi_decode(signal_soft);
    if (signal_bits.size() < 24) return result;
    unsigned rate_code = 0;
    for (unsigned bit = 0; bit < 4; ++bit) rate_code = (rate_code << 1) | signal_bits[bit];
    if (rate_code != 0x0d || signal_bits[4]) return result;
    std::size_t length = 0;
    for (unsigned bit = 0; bit < 12; ++bit) length |= static_cast<std::size_t>(signal_bits[5 + bit]) << bit;
    if (length != kPsduBytes || length > maximum_psdu) return result;
    unsigned parity_value = 0;
    for (unsigned bit = 0; bit < 18; ++bit) parity_value ^= signal_bits[bit];
    if (parity_value) return result;
    for (unsigned bit = 18; bit < 24; ++bit) if (signal_bits[bit]) return result;

    Bits coded;
    for (std::size_t symbol = 0; symbol < kDataSymbols; ++symbol) {
        const auto equalized = equalize(samples + start + 400 + symbol * SYM_SAMP + CP_LEN,
                                        result.csi, cfo, symbol + 1, result.noise_var);
        Bits interleaved;
        for (const auto sc : kDataSc) {
            const auto bits = demap_qam(equalized[sc_to_fft(sc)], 1);
            interleaved.insert(interleaved.end(), bits.begin(), bits.end());
        }
        const auto block = deinterleave(interleaved, 1);
        coded.insert(coded.end(), block.begin(), block.end());
    }
    const auto decoded = viterbi_decode(std::vector<std::int8_t>(coded.begin(), coded.end()));
    if (decoded.size() < 16 + kPsduBytes * 8 + 6) return result;
    const auto descrambled = scramble(decoded);
    const Bits payload_bits(descrambled.begin() + 16,
                            descrambled.begin() + static_cast<std::ptrdiff_t>(16 + kPsduBytes * 8));
    const auto payload = bits_to_bytes(payload_bits);
    constexpr std::size_t fcs_offset = kPsduBytes - 4;
    if (payload.size() != kPsduBytes ||
        payload[0] != 0x08 || payload[1] != 0x00 ||
        payload[kMacHeaderBytes] != 0xaa || payload[kMacHeaderBytes + 1] != 0xaa ||
        payload[kControlOffset - 2] != 0x88 || payload[kControlOffset - 1] != 0xb5 ||
        crc32(payload.data(), fcs_offset) !=
        (static_cast<std::uint32_t>(payload[fcs_offset]) |
         (static_cast<std::uint32_t>(payload[fcs_offset + 1]) << 8) |
         (static_cast<std::uint32_t>(payload[fcs_offset + 2]) << 16) |
         (static_cast<std::uint32_t>(payload[fcs_offset + 3]) << 24))) return result;
    std::copy(payload.begin() + static_cast<std::ptrdiff_t>(kControlOffset),
              payload.begin() + static_cast<std::ptrdiff_t>(kControlOffset + kMetadataBytes),
              result.metadata.begin());
    result.frame_samples = kFrameSamples;
    result.ok = true;
    return result;
}

} // namespace ofdm::wifi_anchor
