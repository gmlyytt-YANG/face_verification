cmake_minimum_required(VERSION 3.5)
project(face_verification)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_COMPILER "g++")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
SET(CMAKE_BUILD_TYPE Debug)

INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


SET(src face_verification.cpp)
SET(hdrs include/face_verification.h)

INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(/usr/local/cuda/include)

LINK_DIRECTORIES(lib_base)

ADD_EXECUTABLE(face_verification main.cpp face_verification.cpp)

TARGET_LINK_LIBRARIES(face_verification caffe glog boost_system pthread protobuf opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
ADD_LIBRARY(fv SHARED ${src} ${hdrs})
