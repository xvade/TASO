cmake_minimum_required(VERSION 3.2)

# Read config.cmake (for USE_CUDA and friends) *before* project(), because
# project(... LANGUAGES CUDA) fails immediately if no CUDA compiler is
# present -- even if USE_CUDA is later set to OFF. This lets us only
# request the CUDA language when we're actually going to use it.
if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include (${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

if (USE_CUDA STREQUAL "ON" OR IS_DIRECTORY "${USE_CUDA}")
  set(TASO_USE_CUDA TRUE)
  project(TASO LANGUAGES CXX CUDA)
else()
  set(TASO_USE_CUDA FALSE)
  project(TASO LANGUAGES CXX)
endif()

#include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(${CMAKE_BINARY_DIR}) # to include protobuf headeer files
include_directories("include")
#include_directories("build")
#include_directories("src/core")

#initial variables
set(TASO_LIBS "")
set(TASO_LINK_LIBS ${CMAKE_DL_LIBS})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Generate protobuf files
find_package(Protobuf REQUIRED)
message(STATUS "PROTOBUF=" ${PROTOBUF_LIBRARY})
include_directories(${PROTOBUF_INCLUDE_DIR})
protobuf_generate_cpp(PROTO_CPPS PROTO_HDRS src/core/rules.proto)
list(APPEND TASO_LINK_LIBS ${PROTOBUF_LIBRARY})

file(GLOB_RECURSE TASO_SRCS
  src/core/*.cc
  )
list(APPEND TASO_SRCS ${PROTO_CPPS})

if (TASO_USE_CUDA)
  file(GLOB_RECURSE TASO_CUDA_SRCS
    src/cudnn/*.cu
    )
  list(APPEND TASO_SRCS ${TASO_CUDA_SRCS})
else()
  # No CUDA toolchain: build against the CPU-only Model/cost-model/
  # execution-stub implementations in src/cpu/ instead of src/cudnn/*.cu.
  # See src/cpu/*.cc for what this trades away (real op execution and
  # hardware-measured costs) and why that's fine for TENSAT's optimizer.
  message(STATUS "TASO_USE_CUDA=OFF: building CPU-only (src/cpu/*.cc), "
                  "with a synthetic cost model instead of real cudnn/cuda "
                  "measurement. Set USE_CUDA=ON in config.cmake to switch "
                  "back to real GPU execution and measurement.")
  file(GLOB_RECURSE TASO_CPU_SRCS
    src/cpu/*.cc
    )
  list(APPEND TASO_SRCS ${TASO_CPU_SRCS})
endif()

#Generic compilation options
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" SUPPORT_CXX11)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("Build in Debug mode")
  set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
endif()

#set CUDA
include(cmake/FindCUDA.cmake)
find_cuda(${USE_CUDA})
if (CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
  message(STATUS "CUDA_INCLUDE_DIR=" ${CUDA_INCLUDE_DIRS})
  add_definitions(-DUSE_CUDNN)
  list(APPEND TASO_LINK_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND TASO_LINK_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND TASO_LINK_LIBS ${CUDA_CUDNN_LIBRARY})
  list(APPEND TASO_LINK_LIBS ${CUDA_CUBLAS_LIBRARY})
elseif(CUDA_FOUND)
  message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
endif(CUDA_FOUND)

add_library(taso_runtime SHARED ${TASO_SRCS})

if (TASO_USE_CUDA)
  set_target_properties(taso_runtime
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# cxx_std_11 (TASO's original setting) doesn't compile against modern
# protobuf/abseil headers (they use C++17 features like std::is_same_v).
# This only affects the C++ language level TASO itself is compiled with,
# not the CPU-vs-CUDA cost model decision above.
target_compile_features(taso_runtime PUBLIC cxx_std_17)

target_link_libraries(taso_runtime ${TASO_LINK_LIBS})

target_include_directories(taso_runtime
  PUBLIC ${PROJECT_SOURCE_DIR}/include)

#install library
install(TARGETS taso_runtime
    LIBRARY DESTINATION lib)

install (DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)
