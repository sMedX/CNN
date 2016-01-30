cmake_minimum_required (VERSION 2.8.11)
project(cutImageByTiles)

include_directories(${AGTK_INCLUDE_DIRS})
list(APPEND LIBRARIES ${AGTK_LIBRARIES})

#
set(SOURCES "cutImageByTiles.cpp")

add_executable(cutImageByTiles ${SOURCES})

TARGET_LINK_LIBRARIES(cutImageByTiles ${LIBRARIES} )