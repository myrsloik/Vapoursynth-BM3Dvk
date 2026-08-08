# VapourSynth-BM3DVulkan

Copyright© 2021 WolframRhodium

Copyright© 2025 Sunflower Dolls

BM3D denoising filter for VapourSynth, running on the core's Vulkan device.

Ported from [VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA),
by way of the Metal implementation this repository previously held.

## Description

- Please check [VapourSynth-BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D) for the original CPU implementation.

- This implementation is built on VapourSynth's Vulkan GPU API (API 4.3). Frames stay in
  video memory for the whole chain, so a `BM3D` → `VAggregate` sequence never crosses the
  PCIe bus in between, and BM3D composes with every other GPU resident filter without a
  transfer.

- The core owns the Vulkan device: it is created once and shared, so there is no per-plugin
  device selection, no separate GPU memory budget, and no duplicate driver initialisation.

## Requirements

- VapourSynth R80 or later, with a GPU and driver that support Vulkan 1.4. Check with
  `core.vulkan_devices` from Python.

- A device with 32- or 64-wide subgroups, which covers every current desktop GPU.

- Works on Linux, Windows and macOS (the latter through MoltenVK).

## Installation

```bash
python3 -m pip install -U vapoursynth vapoursynth-bm3dvulkan
vapoursynth config
```

The wheel installs the plugin into VapourSynth's Python package plugin directory together
with a `manifest.vs` for autoloading.

## Parameters

```python3
bm3dvulkan.BM3D(clip clip[, clip ref=None, float[] sigma=3.0, int[] block_step=8, int[] bm_range=9, int radius=0, int[] ps_num=2, int[] ps_range=4, bint chroma=False, int extractor_exp=0])
```

- clip:

    The input clip. Must be of 32 bit float format. Each plane is denoised separately if `chroma` is set to `False`. Frame properties of the output clip are copied from it.

    A CPU clip is uploaded automatically; pass `core.std.GPUUpload(clip)` explicitly to choose where in the graph the transfer happens.

- ref:

    The reference clip. Must be of the same format, width, height, number of frames as `clip`.

    Used in block-matching and as the reference in empirical Wiener filtering, i.e. `bm3d.Final` / `bm3d.VFinal`:

    ```python3
    basic = core.bm3dvulkan.BM3D(src, radius=0)
    final = core.bm3dvulkan.BM3D(src, ref=basic, radius=0)

    vbasic = core.bm3dvulkan.BM3D(src, radius=radius_nonzero).bm3dvulkan.VAggregate(src=src, planes=[0,1,2])
    vfinal = core.bm3dvulkan.BM3D(src, ref=vbasic, radius=r).bm3dvulkan.VAggregate(src=src, planes=[0,1,2])

    # alternatively, using the v2 interface
    basic_or_vbasic = core.bm3dvulkan.BM3Dv2(src, radius=r)
    final_or_vfinal = core.bm3dvulkan.BM3Dv2(src, ref=basic_or_vbasic, radius=r)
    ```

    corresponds to the followings (ignoring color space handling and other differences in implementation), respectively

    ```python3
    basic = core.bm3d.Basic(clip)
    final = core.bm3d.Final(basic, ref=basic)

    vbasic = core.bm3d.VBasic(src, radius=r).bm3d.VAggregate(radius=r, sample=1)
    vfinal = core.bm3d.VFinal(src, ref=vbasic, radius=r).bm3d.VAggregate(radius=r)
    ```

- sigma:
    The strength of denoising for each plane, defined as the standard deviation of the AWGN component, implying $Noise \sim \mathcal{N}(0, \sigma^2)$ on a $0\text{-}255$ scale.

    The strength is similar (but not strictly equal) as `VapourSynth-BM3D` due to differences in implementation. (coefficient normalization is not implemented, for example)

    A plane whose sigma is zero is not denoised. In spatial mode (`radius=0`) such planes are passed through from the source without a copy; in temporal mode they are zero filled, since the stacked output has no plane of the source's shape to share.

    Default `[3,3,3]`.

- block_step, bm_range, radius, ps_num, ps_range:

    Same as those in `VapourSynth-BM3D`.

    If `chroma` is set to `True`, only the first value is in effect.

    Otherwise an array of values may be specified for each plane (except `radius`).

    `radius` is limited to 15.

    **Note**: It is generally not recommended to take a large value of `ps_num` as current implementations do not take duplicate block-matching candidates into account during temporary searching, which may leads to regression in denoising quality. This issue is not present in `VapourSynth-BM3D`.

    **Note2**: Lowering the value of "block_step" will be useful in reducing blocking artifacts at the cost of slower processing.

- chroma:

    CBM3D algorithm. `clip` must be of `YUV444PS` format.

    Y channel is used in block-matching of chroma channels.

    Default `False`.

- extractor_exp:

    Used for deterministic (bitwise) output.

    [Pre-rounding](https://ieeexplore.ieee.org/document/6545904) is employed for associative floating-point summation.

    The value should be a positive integer not less than 3, and may need to be higher depending on the source video and filter parameters.

    Default `0`. (non-determinism)

## Differences from the Metal implementation

- `device_id` is gone: the core selects and owns the Vulkan device. Use
  `core.set_vulkan_device()` to choose one.

- `fast` is gone. It sized a pool of duplicate resources for concurrent frames; the core's
  execution pool provides that automatically, with the memory scaling to frames actually in
  flight rather than a fixed multiple.

- `zero_init` is gone. Unprocessed planes of the stacked temporal output are always zeroed;
  the flag only ever skipped that, leaving uninitialised data behind.

- `VAggregate` runs on the GPU rather than on the CPU, and takes `src` and `planes`
  arguments. `BM3Dv2` wires it up for you.

## Notes

- `VAggregate` should be called after temporal filtering, as in `VapourSynth-BM3D`.
  Alternatively, use the `BM3Dv2()` interface for both spatial and temporal denoising in one
  step.

- The kernel is one GLSL compute shader specialized at pipeline creation for the temporal,
  chroma and Wiener variants, and for the device's subgroup width.

## Statistics

GPU memory consumption, per frame in flight:

`(ref ? 2 : 1) * (chroma ? 3 : 1) * (2 * radius + 1) * size_of_a_single_frame` for the source stack,
plus `(chroma ? 3 : 1) * (2 * radius + 1) * 2 * size_of_a_single_frame` for the accumulator.

Both come from the core's pooled allocator, count against the VRAM limit
(`core.max_vram_cache_size`) and are recycled between frames.

Compute complexity:

`(chroma ? 3 : 1) * ceil((width - 8) / block_step + 1) * ceil((height - 8) / block_step + 1) * ((2 * bm_range + 1) * (2 * bm_range + 1) + 2 * radius * ps_num * (2 * ps_range + 1) * (2 * ps_range + 1)) * (ref ? 1.5 : 1) + (radius > 0 ? width * height * (chroma ? 3 : 1) * 2 * radius : 0)`

## Compilation

Requires CMake 3.20 or later, a C++20 compiler, VapourSynth's Python package (R80+, for its
headers) and the Vulkan headers. Only the headers are needed — the plugin does not link
against the Vulkan loader, because the core hands it every entry point already resolved.

```bash
python3 -m pip install -U "VapourSynth>=80"
```

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The compiled plugin (`libbm3dvulkan.so`, `bm3dvulkan.dll` or `libbm3dvulkan.dylib`) lands in
`build/lib`. Copy it to your VapourSynth plugins directory.

If CMake cannot locate the headers automatically, pass
`-D VAPOURSYNTH_INCLUDE_DIRECTORY=/path/to/vapoursynth/include` and/or
`-D VULKAN_INCLUDE_DIRECTORY=/path/to/vulkan/include`.

The kernel is compiled into the binary with C23 `#embed` where the compiler supports it
(clang 19+, gcc 15+) and through a CMake-generated header otherwise, so MSVC and older gcc
build without a separate step. Define `BM3DVK_NO_EMBED` to force the generated header.

## License

This project is licensed under the GNU General Public License v3.0 or later (GPLv3+).

Based on [VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA) by WolframRhodium, which is licensed under GPLv2 or later.

The DCT kernels derive from [FFTW](https://www.fftw.org/) generated code, Copyright© 2003, 2007-14 Matteo Frigo and the Massachusetts Institute of Technology.
