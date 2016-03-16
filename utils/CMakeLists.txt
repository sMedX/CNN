cmake_minimum_required (VERSION 2.8.11)
project(cutImageByTiles)

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
project(createRectangleMaskAroundMask)

set(SOURCES "createRectangleMaskAroundMask.cpp")

add_executable(createRectangleMaskAroundMask ${SOURCES})

TARGET_LINK_LIBRARIES(createRectangleMaskAroundMask ${LIBRARIES} )
#
set(SOURCES "cutImageBySlices.cpp" "preprocess.h")

add_executable(cutImageBySlices ${SOURCES})

TARGET_LINK_LIBRARIES(cutImageBySlices ${LIBRARIES} )
