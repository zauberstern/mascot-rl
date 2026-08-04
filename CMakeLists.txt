cmake_minimum_required(VERSION 3.18)
project(volsurf_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-O3 -march=znver1 -mavx2 -mfma -fopenmp -ffast-math)
add_link_options(-fopenmp)

find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Prefer building via setup_extensions.py / torch.utils.cpp_extension.
# This CMakeLists documents the mandated Zen+/OpenMP flags for Layer 1.
add_library(rbergomi_obj OBJECT
  src/engine/rbergomi_engine.cpp
  src/engine/circulant_fft.cpp
  src/engine/dupire_pde.cpp
  src/engine/bs_iv.cpp
  src/engine/heston_cf.cpp
  src/engine/sabr_hagan.cpp
  src/engine/garch_duan.cpp
  src/engine/worlds.cpp
)
target_include_directories(rbergomi_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/engine
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(rbergomi_obj PUBLIC OpenMP::OpenMP_CXX)

message(STATUS "MascotRL: use `python setup_extensions.py` to build pybind/torch modules.")
