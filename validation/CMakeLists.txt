cmake_minimum_required (VERSION 2.8.11)
project(validation)

include_directories(${AGTK_INCLUDE_DIRS})
list(APPEND LIBRARIES ${AGTK_LIBRARIES})

set(SOURCES "validation.cpp")

add_executable(validation ${SOURCES})

TARGET_LINK_LIBRARIES(validation ${LIBRARIES} )

###

set(SOURCES "validationRetVOE.cpp")
add_executable(validationRetVOE ${SOURCES})
TARGET_LINK_LIBRARIES(validationRetVOE ${LIBRARIES} )
