#include "test_framework.h"

extern void run_dsp_tests();
extern void run_txrx_tests();

int main() {
    std::cerr << "\n=== OFDM System Tests ===\n" << std::endl;

    std::cerr << "--- DSP Core Tests ---\n" << std::endl;
    run_dsp_tests();

    std::cerr << "\n--- TX/RX Integration Tests ---\n" << std::endl;
    run_txrx_tests();

    return test_summary();
}
