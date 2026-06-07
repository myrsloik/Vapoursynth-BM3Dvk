cmake_minimum_required(VERSION 3.20)
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0")
project(VapourSynth-BM3DMetal LANGUAGES OBJCXX)

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

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(lib)
