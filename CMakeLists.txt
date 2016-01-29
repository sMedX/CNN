cmake_minimum_required (VERSION 2.8.11)

project(for-caffe)

find_package(AGTK)
include(${AGTK_USE_FILE})

message(STATUS "AGTK_INCLUDE_DIRS")
message(STATUS ${AGTK_INCLUDE_DIRS})

message(STATUS "AGTK_LIBRARIES")
message(STATUS ${AGTK_LIBRARIES})

add_subdirectory(validation)
add_subdirectory(postprocessing)

add_custom_target(for-caffe)