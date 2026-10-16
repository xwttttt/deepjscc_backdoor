cmake_minimum_required(VERSION 3.20)
project(ofdm_system LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
include(third_party/json.cmake)

# Resolve the SDR runtime through the host's pkg-config installation.  Using
# a Conda-provided SoapySDR CMake package here can embed a Conda RUNPATH into
# the native service while the hardware modules (e.g. UHD) are installed in
# the system SoapySDR module directory.  That combination makes
# SoapySDR::Device::make() report "no match" even though SoapySDRUtil sees the
# device.  A pkg-config target keeps the core library and its module search
# paths on the same system installation.
pkg_check_modules(SOAPYSDR REQUIRED IMPORTED_TARGET SoapySDR)
pkg_check_modules(ZMQ QUIET IMPORTED_TARGET libzmq)

if(ZMQ_FOUND)
    set(OFDM_ZMQ_TARGET PkgConfig::ZMQ)
else()
    # Some deployment images ship libzmq.so.5 and the headers through Conda
    # but do not install libzmq.pc.  Prefer the system runtime (which is what
    # the dynamic loader will use) and use the available header location only
    # for compilation.  This fallback must not add a Conda RUNPATH, otherwise
    # it could shadow the system SoapySDR core selected above.
    find_path(ZMQ_INCLUDE_DIR NAMES zmq.h
        PATHS /usr/include /usr/local/include
              /home/gxhen/miniconda3/envs/gnuradio/include
              /home/gxhen/miniconda3/include)
    # A runtime-only installation may expose libzmq.so.5 but not the
    # development symlink libzmq.so. Prefer that exact system SONAME before
    # considering any Conda development library.
    find_file(ZMQ_RUNTIME NAMES libzmq.so.5
        PATHS /lib/${CMAKE_LIBRARY_ARCHITECTURE}
              /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
              /usr/local/lib
        NO_DEFAULT_PATH)
    if(ZMQ_RUNTIME)
        set(ZMQ_LIBRARY "${ZMQ_RUNTIME}")
    else()
        find_library(ZMQ_LIBRARY NAMES zmq
            PATHS /usr/local/lib
                  /home/gxhen/miniconda3/envs/gnuradio/lib
                  /home/gxhen/miniconda3/lib)
    endif()
    if(NOT ZMQ_INCLUDE_DIR OR NOT ZMQ_LIBRARY)
        message(FATAL_ERROR "libzmq headers/runtime not found")
    endif()
    add_library(ofdm_zmq UNKNOWN IMPORTED)
    set_target_properties(ofdm_zmq PROPERTIES
        IMPORTED_LOCATION "${ZMQ_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZMQ_INCLUDE_DIR}")
    set(OFDM_ZMQ_TARGET ofdm_zmq)
    message(STATUS "libzmq pkg-config metadata not found; using ${ZMQ_LIBRARY}")
endif()

# Find liquid-dsp
find_library(LIQUID_LIB liquid PATHS /usr/local/lib /usr/lib)
find_path(LIQUID_INCLUDE liquid.h PATHS /usr/local/include/liquid /usr/local/include /usr/include)

if(NOT LIQUID_LIB)
    message(FATAL_ERROR "liquid-dsp not found")
endif()

message(STATUS "liquid-dsp: ${LIQUID_LIB} (include: ${LIQUID_INCLUDE})")
message(STATUS "SoapySDR:  ${SOAPYSDR_VERSION} (${SOAPYSDR_LIBRARY_DIRS})")
if(ZMQ_FOUND)
    message(STATUS "libzmq:    ${ZMQ_VERSION} (${ZMQ_LIBRARY_DIRS})")
endif()

set(OFDM_SOURCES
    src/dsp/ofdm_core.cc
    src/dsp/preamble.cc
    src/dsp/subcarrier_map.cc
    src/dsp/crc16.cc
    src/dsp/control_header.cc
    src/wifi/wifi_anchor.cc
    src/tx/tx_pipeline.cc
    src/rx/rx_pipeline.cc
    src/interface/zmq_interface.cc
    src/loopback/loopback.cc
    src/radio/soapy_radio.cc
    src/interface/telemetry.cc
)

add_library(ofdm_core STATIC ${OFDM_SOURCES})
target_include_directories(ofdm_core PUBLIC src ${LIQUID_INCLUDE})
target_link_libraries(ofdm_core PUBLIC ${LIQUID_LIB} ${OFDM_ZMQ_TARGET} PkgConfig::SOAPYSDR Threads::Threads m)
target_compile_options(ofdm_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(ofdm-tx-service src/main_tx.cc)
target_link_libraries(ofdm-tx-service PRIVATE ofdm_core)

add_executable(ofdm-rx-service src/main_rx.cc)
target_link_libraries(ofdm-rx-service PRIVATE ofdm_core)

add_executable(ofdm-workbench-service src/main_workbench.cc)
target_link_libraries(ofdm-workbench-service PRIVATE ofdm_core)

add_executable(ofdm-device-probe src/main_device_probe.cc src/radio/device_manager.cc)
target_include_directories(ofdm-device-probe PRIVATE src)
target_compile_definitions(ofdm-device-probe PRIVATE OFDM_HAVE_SOAPY=1)
target_link_libraries(ofdm-device-probe PRIVATE PkgConfig::SOAPYSDR nlohmann_json::nlohmann_json)

# Tests
enable_testing()
add_executable(ofdm_tests tests/test_main.cc tests/test_dsp.cc tests/test_txrx.cc)
target_link_libraries(ofdm_tests PRIVATE ofdm_core)
target_include_directories(ofdm_tests PRIVATE src)
add_test(NAME ofdm_tests COMMAND ofdm_tests)

install(TARGETS ofdm-tx-service ofdm-rx-service ofdm-workbench-service ofdm-device-probe RUNTIME DESTINATION bin)
