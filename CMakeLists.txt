cmake_minimum_required (VERSION 2.8.11)

project(for-caffe)
# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
# ITK
find_package(ITK 4.7 REQUIRED)
include(${ITK_USE_FILE})
list(APPEND LIBRARIES ${ITK_LIBRARIES})

# ------------------------------------------------------------------------------
# AGTK
# ------------------------------------------------------------------------------
find_package(AGTK)
include(${AGTK_USE_FILE})
list(APPEND LIBRARIES ${AGTK_LIBRARIES})

#set(AGTK_DIR  CACHE PATH "AGTK sources ")

#message(STATUS "AGTK_DIR")
#message(STATUS ${AGTK_DIR})

#set(INCLUDE_DIRS "${AGTK_DIR}/Modules/Core")

# ------------------------------------------------------------------------------
# CUDA
# ------------------------------------------------------------------------------
include(FindCUDA)
find_package(CUDA REQUIRED)
#TODO ADD

# ------------------------------------------------------------------------------
# caffe
# ------------------------------------------------------------------------------
#find_package(Caffe)
#include_directories(${Caffe_INCLUDE_DIRS})
#add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

#list(APPEND LIBRARIES ${Caffe_LIBRARIES})

#
message(STATUS "INCLUDE_DIRS ${INCLUDE_DIRS}")

message(STATUS "LIBRARIES ${LIBRARIES}")

add_subdirectory(validation)
add_subdirectory(postprocessing)
add_subdirectory(utils)

#
include_directories(${INCLUDE_DIRS})

set(SOURCES "classification.cpp")

add_executable(classification ${SOURCES})
TARGET_LINK_LIBRARIES(classification ${LIBRARIES} )

#
add_custom_target(for-caffe)