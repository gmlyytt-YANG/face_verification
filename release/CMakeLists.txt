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

SET(CAFFE_GLOG_DIR /usr/local/lib)
SET(BOOST_SYSTEM_PROTOBUF_DIR /usr/lib/x86_64-linux-gnu)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/usr/local/cuda/include)

ADD_EXECUTABLE(face_verification main.cpp face_verification.h face_verification.cpp)

LINK_DIRECTORIES(${CAFFE_GLOG_DIR} ${BOOST_SYSTEM_PROTOBUF_DIR} ${OpenCV_LIBRARY_DIRS})
TARGET_LINK_LIBRARIES(face_verification caffe glog boost_system pthread protobuf ${OpenCV_LIBS})
