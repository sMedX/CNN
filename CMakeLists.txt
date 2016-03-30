cmake_minimum_required (VERSION 2.8.11)

project(for-caffe)

# ------------------------------------------------------------------------------
# C++11 settings
# ------------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-std=c\\+\\+11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

set(AGTK_CXX_STANDARD 11)

set(CMAKE_CXX_EXTENSIONS 0)
set(CMAKE_CXX_STANDARD ${AGTK_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED 1)

find_package(AGTK)
include(${AGTK_USE_FILE})

message(STATUS "AGTK_INCLUDE_DIRS")
message(STATUS ${AGTK_INCLUDE_DIRS})

message(STATUS "AGTK_LIBRARIES")
message(STATUS ${AGTK_LIBRARIES})

add_subdirectory(validation)
add_subdirectory(postprocessing)
add_subdirectory(utils)

add_custom_target(for-caffe)