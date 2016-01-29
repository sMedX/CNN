project(postprocessing)

include_directories(${AGTK_INCLUDE_DIRS})
list(APPEND LIBRARIES ${AGTK_LIBRARIES})

#
set(SOURCES "postprocessing.cpp")

add_executable(postprocessing ${SOURCES})

TARGET_LINK_LIBRARIES(postprocessing ${LIBRARIES} )