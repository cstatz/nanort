set(BUILD_TARGET "benchmark")

include_directories(${CMAKE_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/examples/common")

file(GLOB SOURCES "*.cc" "*.h")
add_executable(${BUILD_TARGET} ${SOURCES})

source_group("Source Files" FILES ${SOURCES})
