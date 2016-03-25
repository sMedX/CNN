cmake_minimum_required (VERSION 2.8.11)
project(cutImageByTiles)

include_directories(${AGTK_INCLUDE_DIRS})
list(APPEND LIBRARIES ${AGTK_LIBRARIES})

#
set(SOURCES "cutImageByTiles.cpp" "preprocess.h")

add_executable(cutImageByTiles ${SOURCES})

TARGET_LINK_LIBRARIES(cutImageByTiles ${LIBRARIES} )

#
project(moveTumorHack)

set(SOURCES "moveTumorHack.cpp")

add_executable(moveTumorHack ${SOURCES})

TARGET_LINK_LIBRARIES(moveTumorHack ${LIBRARIES} )

#
project(dilateMask)

set(SOURCES "dilateMask.cpp")

add_executable(dilateMask ${SOURCES})

TARGET_LINK_LIBRARIES(dilateMask ${LIBRARIES} )
#
set(SOURCES "cutImageBySlices.cpp" "preprocess.h")

add_executable(cutImageBySlices ${SOURCES})

TARGET_LINK_LIBRARIES(cutImageBySlices ${LIBRARIES} )
