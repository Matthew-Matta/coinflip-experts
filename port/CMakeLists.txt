cmake_minimum_required(VERSION 3.16)
project(coinflip_port CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# No fast-math: we need IEEE semantics for faithful rounding emulation.
add_compile_options(-Wall -Wextra -O2 -fno-fast-math -ffp-contract=off)

add_library(coinflip_moe src/moe.cpp)
target_include_directories(coinflip_moe PUBLIC include)

add_executable(test_moe tests/test_moe.cpp)
target_link_libraries(test_moe PRIVATE coinflip_moe)

enable_testing()
add_test(NAME test_moe COMMAND test_moe)
