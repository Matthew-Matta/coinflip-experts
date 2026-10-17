cmake_minimum_required(VERSION 3.16)
project(coinflip_port CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# No fast-math and no implicit FMA contraction: rounding emulation needs IEEE semantics; FMA is used only explicitly.
add_compile_options(-Wall -Wextra -O2 -fno-fast-math -ffp-contract=off)
find_package(Threads REQUIRED)

add_library(coinflip_port src/numerics.cpp src/block.cpp src/io.cpp)
target_include_directories(coinflip_port PUBLIC include)
target_link_libraries(coinflip_port PUBLIC Threads::Threads)

# Commit and dirty state of port/ at build time, recorded in every moe_run manifest (regenerated at each build).
set(BUILD_INFO ${CMAKE_CURRENT_BINARY_DIR}/build_info.cpp)
add_custom_target(build_info
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=${BUILD_INFO} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_info.cmake
  BYPRODUCTS ${BUILD_INFO})

add_executable(moe_run tools/moe_run.cpp ${BUILD_INFO})
add_dependencies(moe_run build_info)
target_link_libraries(moe_run PRIVATE coinflip_port)

add_executable(bench_simd tools/bench_simd.cpp)
target_link_libraries(bench_simd PRIVATE coinflip_port)

add_executable(test_moe tests/test_moe.cpp)
target_link_libraries(test_moe PRIVATE coinflip_port)

enable_testing()
add_test(NAME test_moe COMMAND test_moe)
