cmake_minimum_required (VERSION 2.8.11)

project(for-caffe)
# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
# Find Boost
#set(Boost_USE_STATIC_LIBS ON)
#set(Boost_USE_MULTITHREADED ON)
find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})
list(APPEND LIBRARIES ${Boost_LIBRARIES})

if( WIN32 )
  # disable autolinking in boost
  add_definitions(-DBOOST_ALL_NO_LIB)
endif()

# ITK
find_package(ITK 4.7 REQUIRED)
include(${ITK_USE_FILE})
list(APPEND LIBRARIES ${ITK_LIBRARIES})

# Find OpenCV
find_package(OpenCV 2.4.11 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND LIBRARIES ${OpenCV_LIBRARIES})

# ------------------------------------------------------------------------------
# AGTK
# ------------------------------------------------------------------------------
option(USE_EXTERNAL_AGTK "Use full AGTK" OFF)

if (USE_EXTERNAL_AGTK)
    find_package(AGTK)
    include(${AGTK_USE_FILE})

    include_directories(${AGTK_INCLUDE_DIRS})
    
    list(APPEND LIBRARIES ${AGTK_LIBRARIES})
else()
    include(mini-agtk/CMake/agtkFunctions.cmake)
    
    set(AGTK_Core mini-agtk/Modules/Core)
    set(AGTK_ImageMetrics mini-agtk/Modules/ImageMetrics)
    #${} 
    add_subdirectory(${AGTK_Core})
    add_subdirectory(${AGTK_ImageMetrics})
    
    #list(APPEND INCLUDE_DIRS ${AGTK_Core} ${AGTK_ImageMetrics})
    include_directories(${AGTK_Core} ${AGTK_ImageMetrics})
endif()


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