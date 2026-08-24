cmake_minimum_required(VERSION 3.20)

project(hnswlib
    VERSION 1.0 
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_library(hnsw INTERFACE)

target_include_directories(hnsw
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(node_test
    tests/NodeTest.cpp
    src/Node.cpp
)

target_link_libraries(node_test
    PRIVATE
        hnsw
)

add_test(
    NAME NodeTest 
    COMMAND node_test
)

add_executable(metric_test
    tests/MetricTest.cpp
)

target_link_libraries(metric_test
    PRIVATE
        hnsw
)

add_test(
    NAME MetricTest
    COMMAND metric_test
)