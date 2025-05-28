cmake_minimum_required(VERSION 3.0)
project(GASegmentation)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(SOURCES
    src/base.cpp
    src/utils.cpp
    src/ga.cpp
    src/main.cpp)

find_package(OpenCV REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

add_executable(GASegmentation ${SOURCES})

target_link_libraries(GASegmentation ${OpenCV_LIBS})
target_include_directories(GASegmentation PUBLIC ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
