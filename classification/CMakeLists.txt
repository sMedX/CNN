project(classify)

set(SOURCES "classify3d.cpp")
add_executable(classify3d ${SOURCES})
TARGET_LINK_LIBRARIES(classify3d caffefication ${LIBRARIES} )


set(SOURCES "segment3d.cpp")
add_executable(segment3d ${SOURCES})
TARGET_LINK_LIBRARIES(segment3d caffefication ${LIBRARIES} )