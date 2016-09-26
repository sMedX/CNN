cmake_minimum_required (VERSION 2.8.11)

project(classificationDllTest)

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
# AGTK
# ------------------------------------------------------------------------------
find_package(AGTK)
include(${AGTK_USE_FILE})

list(APPEND LIBRARIES ${AGTK_LIBRARIES})
message(STATUS "AGTK_LIBRARIES ${AGTK_LIBRARIES}")

include_directories(${AGTK_INCLUDE_DIRS} "..")

set(SOURCES "classification.cpp")
add_executable(classificationDllTest ${SOURCES})
target_link_libraries(classificationDllTest ${LIBRARIES})