cmake_minimum_required(VERSION 3.20)

project(hnswlib
    VERSION 1.0 
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

find_program(CLANG_TIDY_EXEC
    clang-tidy
)

if(CLANG_TIDY_EXEC)
    set(CMAKE_CXX_CLANG_TIDY 
        ${CLANG_TIDY_EXEC};
    )
else()
    message("clang-tidy not found")
endif()

enable_testing()

add_library(hnsw INTERFACE)

target_include_directories(hnsw
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(recall_benchmark
    benchmarks/RecallBenchmark.cpp
    src/Node.cpp
    src/HnswBruteIndex.cpp
)

target_link_libraries(recall_benchmark
    PRIVATE
        hnsw
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


add_executable(hnsw_test
    tests/HNSWIndexTest.cpp
    src/Node.cpp
)

target_link_libraries(hnsw_test
    PRIVATE
        hnsw
)

add_test(
    NAME HNSWIndexTest
    COMMAND hnsw_test
)