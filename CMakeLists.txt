cmake_minimum_required(VERSION 3.13)
project(cml)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/executable)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/obj)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/obj)

# Collect source files
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.c")

# build executable
set(GNUPLOT_INTERFACE_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/gnuplot_i/src)

add_executable(main
  ${SRC_FILES}
  ${GNUPLOT_INTERFACE_DIRECTORY}/gnuplot_i.c
)

target_include_directories(main PRIVATE
  ${GNUPLOT_INTERFACE_DIRECTORY}
)

target_compile_options(main PRIVATE -Wall -Wextra -Wno-missing-braces -O2 -lm)
target_link_libraries(main m)
