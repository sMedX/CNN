project(classification)

list(APPEND LIBRARIES ${LIBRARIES})

set(SOURCES "classification.cpp")

add_executable(classification ${SOURCES})

TARGET_LINK_LIBRARIES(classification caffefication ${LIBRARIES} )