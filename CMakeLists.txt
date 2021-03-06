cmake_minimum_required(VERSION 2.8)
#Set project name
PROJECT(MinA)
#disable in source builds
SET(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#Set needed dirs
SET(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(RUNTIME_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_BUILD_TYPE Release)
#Set include and source -dirs
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
FILE(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/MinA/*/*.cpp)
MESSAGE(STATUS "added sources:" ${SOURCES})
#set compiler and linker flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g ")
#Create shared lib and install it to lib
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCES})
INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})

#create a Test
ENABLE_TESTING()
LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})
FILE(GLOB TESTS test/*/*)
SET(TESTS "exceptionTest;sanityTest;logTest")
FOREACH( IT ${TESTS})
    MESSAGE(STATUS "IT:" ${IT})
    FILE(GLOB TESTSOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test/${IT}/*.cpp)
    MESSAGE(STATUS "added sources:" ${TESTSOURCES})
    ADD_EXECUTABLE(${IT}.exe ${TESTSOURCES})
    TARGET_LINK_LIBRARIES(${IT}.exe ${PROJECT_NAME} pthread boost_serialization)
    ADD_TEST(${IT} ${IT}.exe)
ENDFOREACH()
