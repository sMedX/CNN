cmake_minimum_required (VERSION 2.8.11)
project(cutImageByTiles)

# ------------------------------------------------------------------------------
# C++11 settings
# ------------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-std=c\\+\\+11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lboost_system")
endif()

set(AGTK_CXX_STANDARD 11)

set(CMAKE_CXX_EXTENSIONS 0)
set(CMAKE_CXX_STANDARD ${AGTK_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED 1)

if( WIN32 )
 #disable default static linking
 add_definitions(-DBOOST_ALL_DYN_LINK) 
endif()


# Boost filesystem
set(Boost_LIBRARY_DIR "" CACHE PATH "")
find_package(Boost 1.54 REQUIRED COMPONENTS system filesystem) #1.54 for 14.04 ubuntu
include_directories(${Boost_INCLUDE_DIRS})
 
# ITK
find_package(ITK)
include(${ITK_USE_FILE})
include_directories(${ITK_INCLUDE_DIRECTORIES})

# ------------------------------------------------------------------------------
# build-in mini-AGTK
# ------------------------------------------------------------------------------
add_subdirectory(mini-AGTK)
include_directories("mini-AGTK")

#
set(SOURCES "cutImageByTiles.cpp" "preprocess.h")
add_executable(cutImageByTiles ${SOURCES})
TARGET_LINK_LIBRARIES(cutImageByTiles mini-AGTK ${LIBRARIES} )

#
set(SOURCES "cutImageBySlices.cpp" "preprocess.h")
add_executable(cutImageBySlices ${SOURCES})
TARGET_LINK_LIBRARIES(cutImageBySlices mini-AGTK ${LIBRARIES})
