cmake_minimum_required(VERSION 3.10)

project(SDR
        LANGUAGES CXX
        VERSION 0.1.0
        DESCRIPTION "High-dimensional Sharpened Dimensionality Reduction (HD-SDR)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        message("Setting GCC compiler flags.")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Og -march=native")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=x86-64")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        message("Setting MSVC compiler flags.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif ()

# add LGC library
add_library(
        LGC
        SHARED
                ${PROJECT_SOURCE_DIR}/lib/LGC/LGC.cpp
)
set_target_properties(
        LGC
        PROPERTIES PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/lib/LGC/LGC.hpp
        WINDOWS_EXPORT_ALL_SYMBOLS 1
)
target_include_directories(
        LGC
        PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# add DR library
add_library(
        DR
        SHARED
                ${PROJECT_SOURCE_DIR}/lib/DR/DR.cpp
                ${PROJECT_SOURCE_DIR}/lib/DR/DR_interface.cpp
)
set_target_properties(
        DR
        PROPERTIES PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/lib/DR/DR.hpp
        WINDOWS_EXPORT_ALL_SYMBOLS 1
)
target_include_directories(
        DR
        PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# add SDR executable
add_executable(SDR main.cpp)
target_include_directories(
        SDR 
        PUBLIC
                ${PROJECT_SOURCE_DIR}/include
                ${PROJECT_SOURCE_DIR}/lib/LGC
                ${PROJECT_SOURCE_DIR}/lib/DR
)
target_link_libraries(
        SDR
        PUBLIC
                LGC
                DR
)

# install shared libraries in pySDR source directory
set(python_source_dir ${PROJECT_SOURCE_DIR}/python/pySDR)
add_custom_command(
        TARGET LGC POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:LGC> ${python_source_dir}
)
add_custom_command(
        TARGET DR POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:DR> ${python_source_dir}
)