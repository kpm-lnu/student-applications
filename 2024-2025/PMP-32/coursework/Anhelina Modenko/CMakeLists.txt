cmake_minimum_required(VERSION 3.27)
project(fem_engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED PATHS /opt/homebrew/Cellar/glfw/3.4/lib/cmake/glfw3)
find_package(CGAL REQUIRED COMPONENTS Core)

file(GLOB IMGUI_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui/backends/imgui_impl_opengl3.cpp
)

# add_subdirectory(mediapipe_face_mesh)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    /opt/homebrew/include
    /opt/homebrew/Cellar/glm/1.0.1/include/glm
    /opt/homebrew/Cellar/glfw/3.4/include
    ${CGAL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui
    /opt/homebrew/Cellar/cgal/6.0.1/include
)


# Source files
file(GLOB_RECURSE SOURCES
    src/*.cc
    src/*.h
    include/*.h
)

add_executable(fem_engine
    main.cpp
    ${SOURCES}
)

target_sources(${PROJECT_NAME} PRIVATE ${IMGUI_SRC})


set(GMP_INCLUDE_DIR "/opt/homebrew/Cellar/gmp/6.3.0/include")
set(GMP_LIBRARY "/opt/homebrew/Cellar/gmp/6.3.0/lib/libgmp.a")

include_directories(${GMP_INCLUDE_DIR})
link_directories("/opt/homebrew/Cellar/gmp/6.3.0/lib")

# Link libraries
target_link_libraries(fem_engine
    PRIVATE
    OpenGL::GL
    glfw
    ${CGAL_LIBRARIES}
    ${CGAL_3RD_PARTY_LIBRARIES}
    ${GMP_LIBRARY}
)

# This ensures we get a nice core profile on macOS
# But these hints are set at runtime in the code below, not in CMake.
