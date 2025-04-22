# This file is part of TinyAD and released under the MIT license.
# Author: Patrick Schmidt

cmake_minimum_required(VERSION 3.10)

project(TinyAD)

# Set c++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen if not already present
if (NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED)
endif()

# Find OpenMP (optional)
if (NOT TARGET OpenMP::OpenMP_CXX)
    if(APPLE)
        # Homebrew installs libomp here (keg-only)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    endif()

    find_package(OpenMP)
endif()

if (TARGET OpenMP::OpenMP_CXX)
    add_compile_options(${OpenMP_CXX_FLAGS})
    add_link_options(${OpenMP_CXX_FLAGS})
else()
    message("WARNING: Building TinyAD without OpenMP!")
endif()

# Output build type
message("Building TinyAD in " ${CMAKE_BUILD_TYPE} " mode")

# Create library target
file(GLOB_RECURSE TINYAD_HEADERS include/*.hh)
add_library(TinyAD INTERFACE)
#target_sources(TinyAD INTERFACE ${TINYAD_HEADERS}) # IDE shows TinyAD headers for each target
add_custom_target(TinyAD-Headers SOURCES ${TINYAD_HEADERS}) # IDE shows TinyAD headers in one place only
target_include_directories(TinyAD INTERFACE include)
target_compile_options(TinyAD INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
target_link_libraries(TinyAD INTERFACE Eigen3::Eigen)
if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(TinyAD INTERFACE OpenMP::OpenMP_CXX)
endif()

# Create unit tests target
option(TINYAD_UNIT_TESTS "" OFF)
if (${TINYAD_UNIT_TESTS})
    enable_testing()

    message("Building TinyAD with unit tests.")

    if (TARGET GTest::gtest_main)
        message("Found existing gtest target.")
    else()
        message("Downloading googletest.")

        # Avoid DOWNLOAD_EXTRACT_TIMESTAMP warning
        if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
            cmake_policy(SET CMP0135 NEW)
        endif()

        # From https://google.github.io/googletest/quickstart-cmake.html
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/6910c9d9165801d8827d628cb72eb7ea9dd538c5.zip
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    file(GLOB_RECURSE TINYAD_TEST_HEADERS tests/*.hh)
    file(GLOB_RECURSE TINYAD_TEST_SOURCES tests/*.cc)
    add_executable(TinyAD-Tests ${TINYAD_TEST_HEADERS} ${TINYAD_TEST_SOURCES})
    add_dependencies(TinyAD-Tests TinyAD)
    target_include_directories(TinyAD-Tests PUBLIC "include")
    target_link_libraries(TinyAD-Tests PRIVATE
        GTest::gtest_main
        TinyAD
    )
    add_test(TinyAD-Tests TinyAD-Tests)
endif()
