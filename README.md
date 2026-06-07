# VapourSynth-BM3DMETAL

Copyright© 2021 WolframRhodium

Copyright© 2025 Sunflower Dolls

BM3D denoising filter for VapourSynth, implemented in Metal.

Ported from [VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA).

## Description

- Please check [VapourSynth-BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D) for the original CPU implementation.

- This Metal implementation leverages Apple's Metal API for GPU acceleration on macOS systems, providing efficient denoising on Apple Silicon and Intel Macs with Metal-capable GPUs.

## Requirements

- macOS 12.0 (Monterey) or later.

- Metal-capable GPU:
  - Apple Silicon (M1, M2, M3, or newer)
  - Intel Mac with Metal-capable GPU (requires SIMD width/thread execution width of 32; older Intel GPUs may not be supported)

- VapourSynth R74 or later on macOS.

## Installation

For VapourSynth R74 or later, install both VapourSynth and this plugin from PyPI:

```bash
python3 -m pip install -U vapoursynth vapoursynth-bm3dmetal
vapoursynth config
```

The wheel installs `libbm3dmetal.dylib` into VapourSynth's Python package plugin
directory and provides a `manifest.vs` for autoloading.

## Parameters

```python3
bm3dmetal.BM3D(clip clip[, clip ref=None, float[] sigma=3.0, int[] block_step=8, int[] bm_range=9, int radius=0, int[] ps_num=2, int[] ps_range=4, bint chroma=False, int device_id=0, bool fast=False])
```

- clip:

    The input clip. Must be of 32 bit float format. Each plane is denoised separately if `chroma` is set to `False`. Data of unprocessed planes is undefined. Frame properties of the output clip are copied from it.

- ref:

    The reference clip. Must be of the same format, width, height, number of frames as `clip`.

    Used in block-matching and as the reference in empirical Wiener filtering, i.e. `bm3d.Final` / `bm3d.VFinal`:

    ```python3
    basic = core.bm3dmetal.BM3D(src, radius=0)
    final = core.bm3dmetal.BM3D(src, ref=basic, radius=0)

    vbasic = core.bm3dmetal.BM3D(src, radius=radius_nonzero).bm3d.VAggregate(radius=radius_nonzero)
    vfinal = core.bm3dmetal.BM3D(src, ref=vbasic, radius=r).bm3d.VAggregate(radius=r)
    
    # alternatively, using the v2 interface
    basic_or_vbasic = core.bm3dmetal.BM3Dv2(src, radius=r)
    final_or_vfinal = core.bm3dmetal.BM3Dv2(src, ref=basic_or_vbasic, radius=r)
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

    Default `[3,3,3]`.

- block_step, bm_range, radius, ps_num, ps_range:

    Same as those in `VapourSynth-BM3D`.

    If `chroma` is set to `True`, only the first value is in effect.

    Otherwise an array of values may be specified for each plane (except `radius`).
    
    **Note**: It is generally not recommended to take a large value of `ps_num` as current implementations do not take duplicate block-matching candidates into account during temporary searching, which may leads to regression in denoising quality. This issue is not present in `VapourSynth-BM3D`.

    **Note2**: Lowering the value of "block_step" will be useful in reducing blocking artifacts at the cost of slower processing.

- chroma:

    CBM3D algorithm. `clip` must be of `YUV444PS` format.

    Y channel is used in block-matching of chroma channels.

    Default `False`.

- device_id:

    Set Metal GPU device to be used (for systems with multiple GPUs).

    Default `0`.

- fast:

    Enables multi-threaded copy between CPU and GPU, consuming 4x more memory.
    
    - **Apple Silicon**: Enabling this option will degrade performance. Keep it disabled.
    - **Intel Mac with AMD discrete GPU**: May (or may not) provide slight performance improvement.

    Default: `False`.

- extractor_exp:

    Used for deterministic (bitwise) output.

    [Pre-rounding](https://ieeexplore.ieee.org/document/6545904) is employed for associative floating-point summation.

    The value should be a positive integer not less than 3, and may need to be higher depending on the source video and filter parameters.

    Default `0`. (non-determinism)

- zero_init:

    This parameter only has an effect in **temporal mode** (`radius > 0`).

    It controls the output of planes that are **not** being processed (i.e., where `sigma` is set to 0).

    - `True` (default): Unprocessed planes will be filled with zeros (resulting in a black image).
    - `False`: Unprocessed planes will contain uninitialized data, which may appear as garbage or random noise.

    This parameter has no effect in spatial mode (`radius = 0`), as unprocessed planes are copied from the source clip. It also has no effect on planes that are actively being denoised.

    Default: `True`.

## Notes

- `bm3d.VAggregate` should be called after temporal filtering, as in `VapourSynth-BM3D`. Alternatively, you may use the `BM3Dv2()` interface for both spatial and temporal denoising in one step.

- The Metal implementation uses Metal Shading Language for GPU kernels, providing efficient computation on Apple platforms.

## Statistics

GPU memory consumptions:

`(ref ? 4 : 3) * (chroma ? 3 : 1) * (fast ? 4 : 1) * (2 * radius + 1) * size_of_a_single_frame`

Compute complexity:

`(chroma ? 3 : 1) * ceil((width - 8) / block_step + 1) * ceil((height - 8) / block_step + 1) * ((2 * bm_range + 1) * (2 * bm_range + 1) + 2 * radius * ps_num * (2 * ps_range + 1) * (2 * ps_range + 1)) * (ref ? 1.5 : 1) + (radius > 0 ? width * height * (chroma ? 3 : 1) * 2 * radius : 0)`

## Benchmarks

input: 1920x1080

- `chroma=False`: `GrayS`
- `chroma=True`: `YUV444PS`

data format: fps

| radius | chroma | final | M2 Pro 32GB (macOS 15.6.1) | M4 16GB (macOS 26.1) |
| ------ | ------ | ----- | -------------------------- | -------------------- |
| 0      | False  | False | 120.31                     | 173.74               |
| 0      | False  | True  | 102.20                     | 102.86               |
| 0      | True   | False | 56.09                      | 71.08                |
| 0      | True   | True  | 48.20                      | 49.18                |
| 1      | False  | False | 63.20                      | 71.56                |
| 1      | False  | True  | 57.61                      | 51.07                |
| 1      | True   | False | 29.03                      | 24.93                |
| 1      | True   | True  | 25.03                      | 25.39                |
| 2      | False  | False | 44.35                      | 58.43                |
| 2      | False  | True  | 39.69                      | 53.94                |
| 2      | True   | False | 20.34                      | 16.46                |
| 2      | True   | True  | 18.46                      | 11.99                |

## Compilation

Requires CMake 3.20 or later, Xcode Command Line Tools, and VapourSynth's Python
package.

```bash
python3 -m pip install -U "VapourSynth>=74"
```

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

The compiled dynamic library (`libbm3dmetal.dylib`) will be located in the `build/lib` directory. Copy it to your VapourSynth plugins directory.

If CMake cannot locate VapourSynth's headers from Python, pass
`-D VAPOURSYNTH_INCLUDE_DIRECTORY="/path/to/vapoursynth/include"`.

## License

This project is licensed under the GNU General Public License v3.0 or later (GPLv3+).

Based on [VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA) by WolframRhodium, which is licensed under GPLv2 or later.
