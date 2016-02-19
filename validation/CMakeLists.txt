cmake_minimum_required (VERSION 2.8.11)
project(validation)

include_directories(${AGTK_INCLUDE_DIRS})
list(APPEND LIBRARIES ${AGTK_LIBRARIES})

#
set(SOURCES "validation.cpp")

add_executable(validation ${SOURCES})

TARGET_LINK_LIBRARIES(validation ${LIBRARIES} )

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