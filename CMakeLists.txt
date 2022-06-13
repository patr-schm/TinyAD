# This file is part of TinyAD and released under the MIT license.
# Author: Patrick Schmidt

cmake_minimum_required(VERSION 3.9)

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
    find_package(OpenMP)
endif()

if (NOT TARGET OpenMP::OpenMP_CXX)
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
target_link_libraries(TinyAD INTERFACE Eigen3::Eigen)
if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(TinyAD INTERFACE OpenMP::OpenMP_CXX)
endif()

# Create unit tests target
option(TINYAD_UNIT_TESTS "" OFF)
if (${TINYAD_UNIT_TESTS})
    enable_testing()

    message("Building TinyAD with unit tests.")

    if (TARGET gtest)
        add_library(GTest::GTest ALIAS gtest)
        add_library(GTest::Main ALIAS gtest_main)
    endif()

    if (NOT TARGET GTest::GTest)
        find_package(GTest REQUIRED)
    endif()

    file(GLOB_RECURSE TINYAD_TEST_HEADERS tests/*.hh)
    file(GLOB_RECURSE TINYAD_TEST_SOURCES tests/*.cc)
    add_executable(TinyAD-Tests ${TINYAD_TEST_HEADERS} ${TINYAD_TEST_SOURCES})
    add_dependencies(TinyAD-Tests TinyAD)
    target_include_directories(TinyAD-Tests PUBLIC "include")
    target_link_libraries(TinyAD-Tests PRIVATE
        GTest::Main
        TinyAD
    )
    add_test(TinyAD-Tests TinyAD-Tests)
endif()
