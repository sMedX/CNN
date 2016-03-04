project(postprocessing)

include_directories(${INCLUDE_DIRS})
list(APPEND LIBRARIES ${LIBRARIES})

#
set(SOURCES "postprocessing.cpp")

add_executable(postprocessing ${SOURCES})

TARGET_LINK_LIBRARIES(postprocessing ${LIBRARIES} )