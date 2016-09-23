cmake_minimum_required (VERSION 2.8.11)
project(mini-AGTK)

# ------------------------------------------------------------------------------
# C++11 settings
# ------------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
  if(NOT CMAKE_CXX_FLAGS MATCHES "-std=c\\+\\+11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lboost_system")
endif()

set(CMAKE_CXX_EXTENSIONS 11)
set(CMAKE_CXX_STANDARD ${AGTK_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED 1)


# Boost filesystem
set(Boost_LIBRARY_DIR "" CACHE PATH "")
find_package(Boost 1.54 REQUIRED COMPONENTS system filesystem) #1.54 for 14.04 ubuntu
include_directories(${Boost_INCLUDE_DIRS})
 
set(SOURCES "agtkCommandLineArgumentParser.cxx" 
"agtkCommandLineArgumentParser.h" 
"agtkCoreExport.h" 
"agtkIO.cxx" 
"agtkIO.h" 
"agtkTypes.h"
"agtkResampling.h")

add_library(mini-AGTK ${SOURCES})

TARGET_LINK_LIBRARIES(mini-AGTK ${ITK_LIBRARIES} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} )
