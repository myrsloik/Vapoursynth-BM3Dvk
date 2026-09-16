cmake_minimum_required(VERSION 3.20)
project(VapourSynth-BM3DVulkan LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")
if(NOT VAPOURSYNTH_INCLUDE_DIRECTORY)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import vapoursynth; print(vapoursynth.get_include(), end='')"
            OUTPUT_VARIABLE VAPOURSYNTH_INCLUDE_DIRECTORY
            ERROR_QUIET
        )
    endif()
endif()

if(NOT VAPOURSYNTH_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "VapourSynth headers were not found. Install VapourSynth from PyPI or pass -DVAPOURSYNTH_INCLUDE_DIRECTORY=/path/to/vapoursynth/include.")
endif()

# VSVulkan4.h ships with VapourSynth from API 4.3 on; without it there is nothing to build
# against, and the failure would otherwise be an unhelpful missing-include error.
if(NOT EXISTS "${VAPOURSYNTH_INCLUDE_DIRECTORY}/VSVulkan4.h")
    message(FATAL_ERROR "VSVulkan4.h was not found in ${VAPOURSYNTH_INCLUDE_DIRECTORY}. This plugin needs VapourSynth API 4.3 or later.")
endif()

# VSVULKANAPI is laid out for reading, not for offsets: R80 added gpuExecWaitValue and
# gpuTimelineWaitValue in the middle of the struct after its first release candidate, and a
# plugin built against the older layout loads fine and calls the neighbouring function. The
# API version macro did not move with it, so the header itself is what says which one it is.
file(STRINGS "${VAPOURSYNTH_INCLUDE_DIRECTORY}/VSVulkan4.h" _vs_release_api REGEX "gpuExecWaitValue")
if(NOT _vs_release_api)
    message(FATAL_ERROR "VSVulkan4.h in ${VAPOURSYNTH_INCLUDE_DIRECTORY} is from an R80 pre-release. This plugin needs the released R80 headers (pip install -U \"VapourSynth>=80\").")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(lib)
