cmake_minimum_required(VERSION 3.20)

# Headers only: the core hands every Vulkan entry point to the plugin already resolved, so
# nothing links against the loader. Prefer the SDK's headers-only target, fall back to a
# plain find_package(Vulkan), then to a cache path.
set(VULKAN_INCLUDE_DIRECTORY "" CACHE PATH "Path to Vulkan headers (only needed if find_package cannot locate them)")
set(_vulkan_headers "")
if(VULKAN_INCLUDE_DIRECTORY)
    set(_vulkan_headers "${VULKAN_INCLUDE_DIRECTORY}")
else()
    find_package(VulkanHeaders QUIET)
    if(NOT TARGET Vulkan::Headers)
        find_package(Vulkan QUIET)
    endif()
    if(TARGET Vulkan::Headers)
        get_target_property(_vulkan_headers Vulkan::Headers INTERFACE_INCLUDE_DIRECTORIES)
    elseif(Vulkan_INCLUDE_DIRS)
        set(_vulkan_headers "${Vulkan_INCLUDE_DIRS}")
    endif()
endif()
if(NOT _vulkan_headers)
    message(FATAL_ERROR "Vulkan headers were not found. Install the Vulkan SDK or your distribution's vulkan-headers package, or pass -DVULKAN_INCLUDE_DIRECTORY=/path/to/include.")
endif()

# A raw string copy of the kernel, for compilers without C23 #embed (MSVC, gcc < 15).
# bm3dvulkan.cpp prefers #embed where the compiler has it and includes this otherwise, so
# it is generated unconditionally and costs nothing when unused; DEPENDS regenerates it
# whenever the shader changes.
set(_shader_header "${CMAKE_CURRENT_BINARY_DIR}/shader_comp.h")
add_custom_command(
    OUTPUT "${_shader_header}"
    COMMAND ${CMAKE_COMMAND}
        -DIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/shader.comp
        -DOUT_FILE=${_shader_header}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/EmbedShader.cmake
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shader.comp"
            "${CMAKE_CURRENT_SOURCE_DIR}/EmbedShader.cmake"
    COMMENT "Generating shader_comp.h from shader.comp"
    VERBATIM
)

add_library(bm3dvulkan SHARED
    bm3dvulkan.cpp
    "${_shader_header}"
)

target_include_directories(bm3dvulkan PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${_vulkan_headers}
    ${CMAKE_CURRENT_SOURCE_DIR}   # gpufilter.h and, for #embed, shader.comp
    ${CMAKE_CURRENT_BINARY_DIR}   # the generated shader_comp.h
)

# gpufilter.h needs C++20.
target_compile_features(bm3dvulkan PRIVATE cxx_std_20)

if(MSVC)
    target_compile_definitions(bm3dvulkan PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_compile_options(bm3dvulkan PRIVATE /bigobj)
else()
    # #embed is a C23 extension in C++ mode; only the compilers that have it see this.
    target_compile_options(bm3dvulkan PRIVATE -Wno-c23-extensions)
endif()

install(TARGETS bm3dvulkan LIBRARY DESTINATION lib RUNTIME DESTINATION lib)
