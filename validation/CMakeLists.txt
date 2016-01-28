cmake_minimum_required (VERSION 2.8.11)
project(Validation)

find_package(AGTK)
include(${AGTK_USE_FILE})

message(STATUS "AGTK_INCLUDE_DIRS")
message(STATUS ${AGTK_INCLUDE_DIRS})

message(STATUS "AGTK_LIBRARIES")
message(STATUS ${AGTK_LIBRARIES})

include_directories(${AGTK_INCLUDE_DIRS})
list(APPEND LIBRARIES ${AGTK_LIBRARIES})

#
set(SOURCES "validation.cpp")

add_executable(Validation ${SOURCES})

TARGET_LINK_LIBRARIES(Validation ${LIBRARIES} )