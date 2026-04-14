# C++ bootstrap: ONNX Runtime + golden NPZ parity (segmentation first chunk).
cmake_minimum_required(VERSION 3.16)
project(pyannote_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ---- cnpy (NPZ / NPY) -------------------------------------------------------
FetchContent_Declare(
  cnpy_external
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG master
)
FetchContent_GetProperties(cnpy_external)
if(NOT cnpy_external_POPULATED)
  FetchContent_Populate(cnpy_external)
endif()

add_library(cnpy_static STATIC "${cnpy_external_SOURCE_DIR}/cnpy.cpp")
target_include_directories(cnpy_static PUBLIC "${cnpy_external_SOURCE_DIR}")
find_package(ZLIB REQUIRED)
target_link_libraries(cnpy_static PUBLIC ZLIB::ZLIB)

# ---- ONNX Runtime (prebuilt SDK) -------------------------------------------
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "onnxruntime root" FORCE)
endif()
if(NOT ONNXRUNTIME_ROOT)
  message(
    FATAL_ERROR
      "Set ONNXRUNTIME_ROOT to a prebuilt ONNX Runtime tree "
      "(folders include/ and lib/ or lib64/), e.g. after unpacking "
      "onnxruntime-osx-arm64-*.tgz. See cpp/README.md."
  )
endif()

find_path(
  ORT_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include/onnxruntime" "${ONNXRUNTIME_ROOT}/include"
  NO_DEFAULT_PATH
)
find_library(
  ORT_LIBRARY
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
  NO_DEFAULT_PATH
)
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(
    FATAL_ERROR
      "Could not find onnxruntime under ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
  )
endif()

# ---- segmentation golden test ---------------------------------------------
add_executable(segmentation_golden_test tests/segmentation_golden_test.cpp)
target_include_directories(segmentation_golden_test PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(
  segmentation_golden_test PRIVATE cnpy_static "${ORT_LIBRARY}"
)
if(APPLE)
  target_link_libraries(segmentation_golden_test PRIVATE "-framework Foundation")
endif()

add_executable(embedding_golden_test tests/embedding_golden_test.cpp)
target_include_directories(embedding_golden_test PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(
  embedding_golden_test PRIVATE cnpy_static "${ORT_LIBRARY}"
)
if(APPLE)
  target_link_libraries(embedding_golden_test PRIVATE "-framework Foundation")
endif()

add_executable(speaker_count_golden_test tests/speaker_count_golden_test.cpp)
target_link_libraries(speaker_count_golden_test PRIVATE cnpy_static)

add_executable(reconstruct_golden_test tests/reconstruct_golden_test.cpp)
target_link_libraries(reconstruct_golden_test PRIVATE cnpy_static)

add_executable(annotation_golden_test tests/annotation_golden_test.cpp)
target_include_directories(annotation_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/port")
target_link_libraries(annotation_golden_test PRIVATE cnpy_static)

add_executable(community1_shortpath tools/community1_shortpath.cpp)
target_include_directories(community1_shortpath PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/port")
target_include_directories(community1_shortpath PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(community1_shortpath PRIVATE cnpy_static "${ORT_LIBRARY}")
if(APPLE)
  target_link_libraries(community1_shortpath PRIVATE "-framework Foundation")
endif()

message(STATUS "ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
message(STATUS "ORT_INCLUDE_DIR=${ORT_INCLUDE_DIR}")
message(STATUS "ORT_LIBRARY=${ORT_LIBRARY}")
