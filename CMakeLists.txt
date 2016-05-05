cmake_minimum_required (VERSION 2.8.11)

project(caffefication)

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

add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES -DNO_STRICT -DNO_VTK)
# ------------------------------------------------------------------------------
# OpenCV
# ------------------------------------------------------------------------------
find_package(OpenCV)
#include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
#list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})

# ------------------------------------------------------------------------------
# Boost
# ------------------------------------------------------------------------------
#find_package(Boost REQUIRED COMPONENTS filesystem)
#include_directories(${Boost_INCLUDE_DIRS})
#set(LIBRARIES ${Boost_LIBRARIES})

#message(STATUS "Boost_LIBRARIES")
#message(STATUS ${Boost_LIBRARIES})

#if( WIN32 )
  # disable autolinking in boost
 # add_definitions(-DBOOST_ALL_NO_LIB)
#endif()

# ------------------------------------------------------------------------------
# AGTK
# ------------------------------------------------------------------------------
find_package(AGTK)
include(${AGTK_USE_FILE})

list(APPEND LIBRARIES ${AGTK_LIBRARIES})
#list(REMOVE_ITEM LIBRARIES "ITKVideoBridgeOpenCV")
#list(REMOVE_ITEM LIBRARIES "ITKVtkGlue")
message(STATUS "AGTK_LIBRARIES ${AGTK_LIBRARIES}")

include_directories(${AGTK_INCLUDE_DIRS})

# ------------------------------------------------------------------------------
# Caffe
# ------------------------------------------------------------------------------
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})

add_definitions(${Caffe_DEFINITIONS})
list(APPEND LIBRARIES ${Caffe_LIBRARIES})

message(STATUS "Caffe_DEFINITIONS ${Caffe_DEFINITIONS}")
message(STATUS "Caffe_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS}")
message(STATUS "Caffe_LIBRARIES ${Caffe_LIBRARIES}")
message(STATUS "Caffe_LINKER_LIBS ${Caffe_LINKER_LIBS}")

add_subdirectory(classification)
add_subdirectory(validation)
add_subdirectory(postprocessing)
add_subdirectory(utils)

#list(APPEND LIBRARIES "D:/alex/caffe-ms/NugetPackages/hdf5-v120-complete.1.8.15.2/lib/native/lib/x64/hdf5_hl.lib") #todo
list(APPEND LIBRARIES "C:/Program Files/HDF_Group/HDF5/1.8.16/lib/hdf5_hl.lib;C:/Program Files/HDF_Group/HDF5/1.8.16/lib/hdf5_hl_cpp.lib;") #todo
#TODO gdcm lib from itk

set(SOURCES "caffefication.cpp" "caffefication.h")
include_directories("utils")
add_library(caffefication SHARED ${SOURCES})
target_link_libraries(caffefication ${LIBRARIES})