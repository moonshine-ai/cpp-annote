# C++ bootstrap: ONNX Runtime + golden NPZ parity (segmentation first chunk).
cmake_minimum_required(VERSION 3.16)
project(pyannote_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

# ---- Eigen (header-only) ----------------------------------------------------
FetchContent_Declare(
  eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_GetProperties(eigen3)
if(NOT eigen3_POPULATED)
  FetchContent_Populate(eigen3)
endif()

# Eigen: only MPL2-licensed modules (dense solvers we use remain available).
add_library(eigen_mpl2 INTERFACE)
target_compile_definitions(eigen_mpl2 INTERFACE EIGEN_MPL2_ONLY=1)
target_include_directories(eigen_mpl2 INTERFACE "${eigen3_SOURCE_DIR}")

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

# ---- kaldi-native-fbank (+ kissfft via upstream cmake) ----------------------
option(PYANNOTE_FETCH_KALDI_NATIVE_FBANK "Fetch and build kaldi-native-fbank + kissfft" ON)
if(PYANNOTE_FETCH_KALDI_NATIVE_FBANK)
  set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    kaldi_native_fbank
    GIT_REPOSITORY https://github.com/csukuangfj/kaldi-native-fbank.git
    GIT_TAG v1.22.3
  )
  FetchContent_MakeAvailable(kaldi_native_fbank)
endif()

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

add_executable(full_segmentation_window_parity_test tests/full_segmentation_window_parity_test.cpp)
target_include_directories(full_segmentation_window_parity_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(full_segmentation_window_parity_test PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(
  full_segmentation_window_parity_test PRIVATE cnpy_static "${ORT_LIBRARY}"
)
if(APPLE)
  target_link_libraries(full_segmentation_window_parity_test PRIVATE "-framework Foundation")
endif()

add_executable(
  embedding_golden_test
  tests/embedding_golden_test.cpp
  src/compute_fbank.cpp
  src/embedding_ort_infer.cpp
)
target_include_directories(embedding_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(embedding_golden_test PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(
  embedding_golden_test PRIVATE cnpy_static "${ORT_LIBRARY}"
)
if(PYANNOTE_FETCH_KALDI_NATIVE_FBANK)
  target_include_directories(embedding_golden_test PRIVATE "${kaldi_native_fbank_SOURCE_DIR}")
  target_link_libraries(embedding_golden_test PRIVATE kaldi-native-fbank-core)
endif()
if(APPLE)
  target_link_libraries(embedding_golden_test PRIVATE "-framework Foundation")
endif()

add_executable(speaker_count_golden_test tests/speaker_count_golden_test.cpp)
target_link_libraries(speaker_count_golden_test PRIVATE cnpy_static)

add_executable(frozen_golden_inputs_test tests/frozen_golden_inputs_test.cpp)
target_compile_definitions(
  frozen_golden_inputs_test
  PRIVATE "PYANNOTE_FROZEN_GOLDEN_UTTERANCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s\""
          "PYANNOTE_FROZEN_GOLDEN_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/golden/callhome_eng_idx0\""
)
target_link_libraries(frozen_golden_inputs_test PRIVATE cnpy_static)

add_executable(reconstruct_golden_test tests/reconstruct_golden_test.cpp)
target_link_libraries(reconstruct_golden_test PRIVATE cnpy_static)

add_executable(annotation_golden_test tests/annotation_golden_test.cpp)
target_include_directories(annotation_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(annotation_golden_test PRIVATE cnpy_static)

add_executable(
  clustering_golden_test
  tests/clustering_golden_test.cpp
  src/scipy_linkage.cpp
  src/filter_train.cpp
  src/plda_vbx.cpp
  src/clustering_vbx.cpp
  src/parity_log.cpp
)
target_include_directories(clustering_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(clustering_golden_test PRIVATE cnpy_static eigen_mpl2)

add_executable(vbx_parity_test tests/vbx_parity_test.cpp src/plda_vbx.cpp)
target_include_directories(vbx_parity_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(vbx_parity_test PRIVATE cnpy_static eigen_mpl2)

add_executable(plda_fea_golden_test tests/plda_fea_golden_test.cpp src/plda_vbx.cpp)
target_include_directories(plda_fea_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(plda_fea_golden_test PRIVATE cnpy_static eigen_mpl2)

add_executable(
  linkage_fcluster_golden_test
  tests/linkage_fcluster_golden_test.cpp
  src/scipy_linkage.cpp
)
target_include_directories(linkage_fcluster_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(linkage_fcluster_golden_test PRIVATE cnpy_static)

add_executable(
  filter_ahc_golden_test
  tests/filter_ahc_golden_test.cpp
  src/filter_train.cpp
  src/scipy_linkage.cpp
)
target_include_directories(filter_ahc_golden_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(filter_ahc_golden_test PRIVATE cnpy_static eigen_mpl2)

add_executable(
  cpp-annote-cli
  tools/cpp-annote-cli.cpp
  src/cpp-annote.cpp
  src/cpp-annote-streaming.cpp
  src/community1_cpp_annote_embedded.cpp
  src/community1_ort_embedded.cpp
  src/compute_fbank.cpp
  src/embedding_ort_infer.cpp
  src/filter_train.cpp
  src/clustering_vbx.cpp
  src/plda_vbx.cpp
  src/scipy_linkage.cpp
  src/parity_log.cpp
)
target_include_directories(cpp-annote-cli PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(cpp-annote-cli PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(cpp-annote-cli PRIVATE cnpy_static "${ORT_LIBRARY}" eigen_mpl2)
if(PYANNOTE_FETCH_KALDI_NATIVE_FBANK)
  target_include_directories(cpp-annote-cli PRIVATE "${kaldi_native_fbank_SOURCE_DIR}")
  target_link_libraries(cpp-annote-cli PRIVATE kaldi-native-fbank-core)
endif()
if(APPLE)
  target_link_libraries(cpp-annote-cli PRIVATE "-framework Foundation")
endif()

if(PYANNOTE_FETCH_KALDI_NATIVE_FBANK)
  add_executable(knf_smoke_test tests/knf_smoke_test.cpp)
  target_include_directories(
    knf_smoke_test PRIVATE "${kaldi_native_fbank_SOURCE_DIR}"
  )
  target_link_libraries(knf_smoke_test PRIVATE kaldi-native-fbank-core)
endif()

message(STATUS "ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
message(STATUS "ORT_INCLUDE_DIR=${ORT_INCLUDE_DIR}")
message(STATUS "ORT_LIBRARY=${ORT_LIBRARY}")
