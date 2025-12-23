cmake_minimum_required(VERSION 3.16)
project(SketchVision)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =================================
# Default build type
# =================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")


# =================================
# Find external packages
# =================================
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)


# =================================
# Sources and includes
# =================================

# Collect all source files
file(GLOB_RECURSE SRC_FILES "src/*.cpp")

# Create executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Add include directory (for your .hpp files)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)


# =================================
# Link libraries
# =================================
target_link_libraries(${PROJECT_NAME}
    gflags
    ${OpenCV_LIBS}
    Threads::Threads
    stdc++fs
    Pcap++
    Packet++
    Common++
    pcap
    pthread
)
