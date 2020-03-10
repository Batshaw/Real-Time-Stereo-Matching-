cmake_minimum_required(VERSION 3.0.2)
project(dense_stereo_matching)

# add cmake flag 
option(ENABLE_KERNEL_PROFILING OFF)

# add preprocessor define based on cmake flag
if(ENABLE_KERNEL_PROFILING)
add_definitions(-DENABLE_KERNEL_PROFILING)
endif()

#set the build type if its not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if (UNIX)
  find_package(PkgConfig)
endif(UNIX)

# Location where cmake first looks for modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
set(GLOBAL_EXT_DIR ${CMAKE_SOURCE_DIR}/external)
include(FindDependency)
include(AddSubLibrary)


################################
#gmp
set(GMP_INCLUDE_SEARCH_DIR "/opt/gmp/include")
set(GMP_LIBRARY_SEARCH_DIR "/opt/gmp/lib")
find_dependency_custom(GMP "gmp.h" "gmp")
################################
#mpfr
set(MPFR_INCLUDE_SEARCH_DIR "/opt/mpfr/include")
set(MPFR_LIBRARY_SEARCH_DIR "/opt/mpfr/lib")
find_dependency_custom(MPFR "mpfr.h" "mpfr")
################################
# glfw
set(GLFW_DIRECTORY glfw-3.2.1)
set(GLFW_INSTALL OFF CACHE STRING "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE STRING "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE STRING "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/${GLFW_DIRECTORY})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/${GLFW_DIRECTORY}/include)

################################
# gapi and libaumer

set(BAUMER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/framework/libBaumer)
set(BAUMER_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/framework/libBaumer)
set(BAUMER_LIBRARIES bgapi)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINUX -D_GNULINUX")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")

include_directories(SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/${BAUMER_INCLUDE_DIR})



set(OPTION_BUILD_TESTS OFF CACHE STRING "" FORCE)
include_directories(SYSTEM external)


################################
# stb
set(STB_DIRECTORY stb)
set(STB_INCLUDE_DIR /external/${STB_DIRECTORY})
include_directories(SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/${STB_INCLUDE_DIR})

set(OPTION_BUILD_TESTS OFF CACHE STRING "" FORCE)
include_directories(SYSTEM external)

# openmp
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()




FIND_PACKAGE( Boost COMPONENTS program_options system thread REQUIRED )

#FIND_PACKAGE(Boost REQUIRED)

#OpenCL
FIND_PACKAGE(OpenCL REQUIRED)

#OpenCV
FIND_PACKAGE(OpenCV REQUIRED)

################################

#GLFW
mark_as_advanced(GLFW_BUILD_DOCS GLFW_BUILD_TESTS GLFW_INSTALL GLFW_BUILD_EXAMPLES
 GLFW_DOCUMENT_INTERNALS GLFW_USE_EGL LIB_SUFFIX BUILD_SHARED_LIBS)

add_definitions(-DGLEW_STATIC)
# to prevent CGAL assertion fail 
add_definitions(-frounding-math)

################################
# activate C++ 14
if(NOT MSVC)
    add_definitions(-std=c++14 -fopenmp)
    # show warnings
    add_definitions(-Wall -Wextra)
    # force linking  with c++14 lib
    if(APPLE)
        set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++0y")
        set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
        add_definitions(-stdlib=libc++)
    endif()
else()
    # build in parallel, show warnings
    add_definitions(/MP /W3)
endif()

# Add output directory
if(MSVC)
  set(BINARY_DIRECTORY build)
endif()
# MSVC & Xcode automatically create the build-type folders
if(MSVC OR CMAKE_GENERATOR STREQUAL Xcode)
  SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
  SET(LIBRARY_OUTPUT_PATH  ${PROJECT_BINARY_DIR})
else()
  SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
  SET(LIBRARY_OUTPUT_PATH  ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endif()

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# framework
add_subdirectory(framework)
# applications
add_subdirectory(applications)

################################
# suppress displaying of external lib options  
#GLFW
mark_as_advanced(GLFW_BUILD_DOCS GLFW_BUILD_TESTS GLFW_INSTALL GLFW_BUILD_EXAMPLES
 GLFW_DOCUMENT_INTERNALS GLFW_USE_EGL LIB_SUFFIX BUILD_SHARED_LIBS)

################################
#if user didnt set install dir, override it and write it to the cache -> Type and description necessary,
#overwrite variable, not just write it to cache
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE STRING "Install path prefix, prepended onto install directories." FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# copy over shaders
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/glsl
                  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
                  ${CMAKE_CURRENT_SOURCE_DIR}/framework/kernels
                  ${CMAKE_CURRENT_SOURCE_DIR}/applications/images
                   
        DESTINATION bin/
        FILES_MATCHING 
        PATTERN "*.fs"
        PATTERN "*.vs"
        PATTERN "*.gs"
        PATTERN "*.glsl"
        PATTERN "*.png"
        PATTERN "*.cl"
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/framework/libBaumer/bsysgige.xml
              ${CMAKE_CURRENT_SOURCE_DIR}/external/baumer_resources/img_proc.so.16
              ${CMAKE_CURRENT_SOURCE_DIR}/external/baumer_resources/img_proc.so.16.0.8596
              ${CMAKE_CURRENT_SOURCE_DIR}/external/baumer_resources/sys_gige.so.16
              ${CMAKE_CURRENT_SOURCE_DIR}/external/baumer_resources/sys_gige.so.16.0.8596
              ${CMAKE_CURRENT_SOURCE_DIR}/framework/libBaumer/libbgapi.so
              ${CMAKE_CURRENT_SOURCE_DIR}/framework/libBaumer/libbgapi.so.16
              ${CMAKE_CURRENT_SOURCE_DIR}/framework/libBaumer/libbgapi.so.16.0.8596
        DESTINATION  bin/)

