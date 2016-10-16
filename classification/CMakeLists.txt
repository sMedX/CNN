project(classify)

set(SOURCES "classify3d.cpp")
add_executable(classify3d ${SOURCES})
TARGET_LINK_LIBRARIES(classify3d caffefication ${LIBRARIES} )