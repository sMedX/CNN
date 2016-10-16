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

if( WIN32 )
  # disable autolinking in boost
 # add_definitions(-DBOOST_ALL_NO_LIB)
 #disable default static linking
 add_definitions(-DBOOST_ALL_DYN_LINK) 
endif()

# ------------------------------------------------------------------------------
# AGTK TODO
# ------------------------------------------------------------------------------

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

set(HDF51_8_16_LIB_DIR "C:/Program Files/HDF_Group/HDF5/1.8.16/lib" CACHE PATH) #todo
#"D:/alex/caffe-ms/NugetPackages/hdf5-v120-complete.1.8.15.2/lib/native/lib/x64/hdf5_hl.lib")
list(APPEND LIBRARIES "${HDF51_8_16_LIB_DIR}/hdf5_hl.lib;${HDF51_8_16_LIB_DIR}/hdf5_hl_cpp.lib;") 

#TODO gdcm lib from itk
set(GDCM_LIB_DIR "D:/apb/SB/MITK-superbuild/ep/lib" CACHE PATH)
list(APPEND LIBRARIES "${GDCM_LIB_DIR}/gdcmMEXD.lib;\
${GDCM_LIB_DIR}/gdcmMSFF.lib;\
${GDCM_LIB_DIR}/gdcmgetopt.lib;\
${GDCM_LIB_DIR}/gdcmopenjpeg.lib;\
${GDCM_LIB_DIR}/gdcmjpeg8.lib;\
${GDCM_LIB_DIR}/gdcmjpeg16.lib;\
${GDCM_LIB_DIR}/gdcmjpeg12.lib;\
${GDCM_LIB_DIR}/gdcmCommon.lib;\
${GDCM_LIB_DIR}/gdcmzlib.lib;\
${GDCM_LIB_DIR}/gdcmDSED.lib;\
${GDCM_LIB_DIR}/gdcmexpat.lib;\
${GDCM_LIB_DIR}/gdcmIOD.lib;\
${GDCM_LIB_DIR}/gdcmDICT.lib;\
${GDCM_LIB_DIR}/gdcmcharls.lib;")

set(SOURCES "caffefication.cpp" "caffefication.h")
include_directories("utils")
add_library(caffefication SHARED ${SOURCES})
target_link_libraries(caffefication ${LIBRARIES})