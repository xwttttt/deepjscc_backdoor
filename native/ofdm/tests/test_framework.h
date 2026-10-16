#pragma once
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

inline int g_tests_run = 0;
inline int g_tests_passed = 0;
inline int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        return false; \
    } \
} while(0)

#define TEST_ASSERT_NEAR(a, b, tol, msg) do { \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "  FAIL: " << msg << " expected=" << (b) << " got=" << (a) << std::endl; \
        return false; \
    } \
} while(0)

#define RUN_TEST(fn) do { \
    g_tests_run++; \
    std::cerr << "Running " << #fn << "... "; \
    if (fn()) { \
        g_tests_passed++; \
        std::cerr << "PASS" << std::endl; \
    } else { \
        g_tests_failed++; \
    } \
} while(0)

inline int test_summary() {
    std::cerr << "\n=== Tests: " << g_tests_passed << "/" << g_tests_run
              << " passed, " << g_tests_failed << " failed ===" << std::endl;
    return g_tests_failed > 0 ? 1 : 0;
}
