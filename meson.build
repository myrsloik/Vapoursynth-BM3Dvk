project('VapourSynth-BM3DVulkan', 'cpp',
    default_options: ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++20'],
    license: 'GPL-3.0-or-later',
    license_files: 'LICENSE',
    meson_version: '>=1.2.3',
    version: '1.0.0',
)

cc = meson.get_compiler('cpp')
py = import('python').find_installation(pure: false)

# Headers only: the core hands the plugin every Vulkan entry point already resolved, so
# nothing links against the loader. The Khronos headers package is looked for first, then the
# loader's package (vulkan.pc, or VULKAN_SDK on Windows) for nothing but its include path, and
# a header on the default search path does too.
vulkan_deps = []
vulkan_headers = dependency('VulkanHeaders', method: 'cmake', modules: ['Vulkan::Headers'], required: false)
if vulkan_headers.found()
    vulkan_deps += vulkan_headers
else
    vulkan_loader = dependency('vulkan', required: false)
    if vulkan_loader.found()
        vulkan_deps += vulkan_loader.partial_dependency(compile_args: true, includes: true)
    endif
endif
if not cc.has_header('vulkan/vulkan.h', dependencies: vulkan_deps)
    error('vulkan/vulkan.h was not found. Install the Vulkan SDK or your distribution\'s vulkan-headers package, or point CMAKE_PREFIX_PATH or VULKAN_SDK at one.')
endif
# VSVulkan4.h is written against Vulkan 1.4: it names vkMapMemory2 and vkUnmapMemory2, which are
# core only from there, and older headers fail deep inside the macro that builds the function
# table without saying why.
if not cc.has_header_symbol('vulkan/vulkan.h', 'VK_API_VERSION_1_4', dependencies: vulkan_deps)
    error('VSVulkan4.h needs Vulkan headers 1.4 or newer, the ones found are older.')
endif

# The API 4.3 GPU declarations, which VapourSynth ships from R80 on; its Python package
# carries the headers.
vs_incdir = include_directories(
    run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip(),
)
if not cc.has_header('VSVulkan4.h', include_directories: vs_incdir, dependencies: vulkan_deps)
    error('VSVulkan4.h was not found in the VapourSynth headers. This plugin needs VapourSynth API 4.3 (R80 or later).')
endif
# VSVULKANAPI is laid out for reading, not for offsets: R80 added gpuExecWaitValue and
# gpuTimelineWaitValue in the middle of the struct after its first release candidate, and a
# plugin built against the older layout loads fine and calls the neighbouring function. The
# API version macro did not move with it, so the header itself is what says which one it is.
if not cc.compiles('''#define VS_USE_API_43
#include "VapourSynth4.h"
#include "VSVulkan4.h"
int probe(const VSVULKANAPI *api) { return api->gpuExecWaitValue != 0; }
''', name: 'released R80 VSVulkan4.h', include_directories: vs_incdir, dependencies: vulkan_deps)
    error('The VapourSynth headers are from an R80 pre-release. This plugin needs the released R80 headers (pip install -U "VapourSynth>=80").')
endif

# The kernel ships inside the binary as a raw string literal, generated from shader.comp so
# the GLSL stays a file glslang can be pointed at.
shader_header = custom_target('shader_comp.h',
    input: 'lib/shader.comp',
    output: 'shader_comp.h',
    command: [py, files('tools/embed_shader.py'), '@INPUT@', '@OUTPUT@'],
)

cpp_args = []
if cc.get_argument_syntax() == 'msvc'
    cpp_args += ['-DNOMINMAX', '-D_CRT_SECURE_NO_WARNINGS', '/bigobj']
endif

# A module rather than a library: nothing links against a plugin. Meson still writes an
# import library beside it on Windows, which pyproject.toml keeps out of the wheel, and gives
# modules a .so suffix on macOS, where the core looks for .dylib.
shared_module('bm3dvulkan',
    files('lib/bm3dvulkan.cpp'),
    shader_header,
    cpp_args: cpp_args,
    dependencies: vulkan_deps,
    include_directories: vs_incdir,
    install: true,
    install_dir: py.get_install_dir() / 'vapoursynth/plugins',
    name_suffix: host_machine.system() == 'darwin' ? 'dylib' : [],
)
