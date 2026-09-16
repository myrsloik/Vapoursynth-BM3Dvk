"""End to end check of the plugin on the core's Vulkan device.

    python test/smoke.py                 # the plugin VapourSynth autoloads
    python test/smoke.py path/to/plugin  # a fresh build, with autoloading disabled so an
                                         # installed copy cannot claim the namespace first

Needs numpy and a GPU the core accepts. Runs in a few seconds: every variant of the kernel
on a constant clip, which must come out constant, and on a noisy gradient, which must come
out closer to the clean signal; the sigma-0 plane paths; VAggregate at the clip edges and
its refusals; and bitwise reproducibility with extractor_exp. Exits non-zero on the first
failure.
"""
import sys

import numpy as np
import vapoursynth as vs


class _NoAutoload(vs.EnvironmentPolicy):
    def __init__(self):
        self._env = None

    def on_policy_registered(self, special_api):
        self._env = special_api.create_environment(vs.DISABLE_AUTO_LOADING)

    def on_policy_cleared(self):
        self._env = None

    def get_current_environment(self):
        return self._env

    def set_environment(self, environment):
        prev = self._env
        self._env = environment
        return prev

    def is_alive(self, environment):
        return True


if len(sys.argv) > 1:
    vs.register_policy(_NoAutoload())
core = vs.core
if len(sys.argv) > 1:
    core.std.LoadPlugin(sys.argv[1])
print("device:", core.vulkan_device_info["name"])

W, H, N = 96, 80, 5
rng = np.random.default_rng(1234)
yy, xx = np.mgrid[0:H, 0:W]
BASE = np.stack([
    0.25 + 0.5 * xx / W,
    0.5 + 0.2 * np.sin(yy / 7.0),
    0.5 - 0.2 * np.cos(xx / 9.0),
]).astype(np.float32)
CLEAN = [BASE + 0.03 * n for n in range(N)]
NOISY = [(c + rng.normal(0, 8.0 / 255.0, c.shape)).astype(np.float32) for c in CLEAN]


def make_clip(arrays):
    blank = core.std.BlankClip(format=vs.YUV444PS, width=W, height=H, length=N, keep=True)

    def fill(n, f):
        fout = f.copy()
        for p in range(3):
            np.asarray(fout[p])[:] = arrays[n][p]
        return fout

    return blank.std.ModifyFrame(blank, fill)


def planes(clip, n):
    """The frame's three planes as one array, downloaded first when it lives on the GPU."""
    if clip.gpu_resident:
        clip = core.std.GPUDownload(clip)
    f = clip.get_frame(n)
    return np.stack([np.asarray(f[p]) for p in range(3)]), f


def mse(a, b):
    return [float(np.mean((a[p] - b[p]) ** 2)) for p in range(3)]


def expect_error(what, fn, text):
    try:
        fn()
    except vs.Error as e:
        assert text in str(e), f"{what}: unexpected error text: {e}"
        return
    raise AssertionError(f"{what}: no error was raised")


src = make_clip(NOISY)
variants = [
    ("spatial", {"radius": 0}),
    ("temporal r1", {"radius": 1}),
    ("temporal r2", {"radius": 2}),
    ("spatial chroma", {"radius": 0, "chroma": True}),
    ("temporal r1 chroma", {"radius": 1, "chroma": True}),
]

# A constant clip has nothing but DC in every block, so it must come back unchanged.
const = core.std.BlankClip(format=vs.YUV444PS, width=W, height=H, length=N, color=[0.5, 0.25, 0.75])
expected = np.array([0.5, 0.25, 0.75], dtype=np.float32)[:, None, None]
for name, kw in variants:
    out, _ = planes(core.bm3dvk.BM3Dv2(const, sigma=3, **kw), 2)
    err = float(np.abs(out - expected).max())
    assert err <= 1e-6, f"constant clip, {name}: max error {err}"
print("constant clips: ok")

# Denoising must move a noisy gradient toward the clean signal, in basic and final alike,
# and a plane with sigma 0 must come through untouched, from the source, in every mode.
for name, kw in variants + [
    ("spatial sigma[0]=0", {"radius": 0, "sigma": [0, 8, 8]}),
    ("temporal sigma[1]=0", {"radius": 1, "sigma": [8, 0, 8]}),
    ("temporal chroma sigma[2]=0", {"radius": 1, "chroma": True, "sigma": [8, 8, 0]}),
]:
    kw = {"sigma": 8, **kw}
    sigma = kw["sigma"] if isinstance(kw["sigma"], list) else [kw["sigma"]] * 3
    basic = core.bm3dvk.BM3Dv2(src, **kw)
    final = core.bm3dvk.BM3Dv2(src, ref=basic, **kw)
    for stage, clip in (("basic", basic), ("final", final)):
        for n in (0, 2, N - 1):
            out, f = planes(clip, n)
            assert np.isfinite(out).all(), f"{name} {stage} frame {n}: non-finite output"
            assert not any(k.startswith("BM3D_V_") for k in f.props), f"{name} {stage}: props leaked"
            before, after = mse(NOISY[n], CLEAN[n]), mse(out, CLEAN[n])
            for p in range(3):
                if sigma[p]:
                    assert after[p] < 0.5 * before[p], \
                        f"{name} {stage} frame {n} plane {p}: mse {after[p]} vs noisy {before[p]}"
                else:
                    assert np.array_equal(out[p], NOISY[n][p]), \
                        f"{name} {stage} frame {n} plane {p}: sigma 0 plane was altered"
    print(f"{name}: ok (frame 2 mse noisy {mse(NOISY[2], CLEAN[2])[0]:.2e} -> {mse(planes(basic, 2)[0], CLEAN[2])[0]:.2e})")

# The stacked intermediate and its explicit aggregation.
stacked = core.bm3dvk.BM3D(src, sigma=[8, 0, 8], radius=1)
assert stacked.height == 6 * H
f = core.std.GPUDownload(stacked).get_frame(0)
assert f.props["BM3D_V_radius"] == 1 and list(f.props["BM3D_V_process"]) == [1, 0, 1], dict(f.props)
agg = core.bm3dvk.VAggregate(stacked, src, planes=[0, 2])
for n in range(N):
    out, f = planes(agg, n)
    assert np.isfinite(out).all(), f"VAggregate frame {n}: non-finite output"
    assert mse(out, CLEAN[n])[0] < 0.5 * mse(NOISY[n], CLEAN[n])[0], f"VAggregate frame {n}: no improvement"
    assert np.array_equal(out[1], NOISY[n][1]), f"VAggregate frame {n}: plane 1 not taken from src"
expect_error("VAggregate of a sigma-0 plane",
    lambda: planes(core.bm3dvk.VAggregate(stacked, src, planes=[0, 1, 2]), 1), "was not denoised")
expect_error("VAggregate plane out of range",
    lambda: core.bm3dvk.VAggregate(stacked, src, planes=[3]), "out of range")
expect_error("VAggregate of a plain clip",
    lambda: core.bm3dvk.VAggregate(src, src, planes=[0]), "radius > 0")
expect_error("negative sigma through BM3Dv2",
    lambda: core.bm3dvk.BM3Dv2(src, sigma=-3), "non-negative")
untouched, _ = planes(core.bm3dvk.BM3Dv2(src, sigma=0, radius=1), 2)
assert np.array_equal(untouched, NOISY[2]), "all-zero sigma did not return the input"
print("VAggregate and argument checks: ok")

# With the extractor the accumulation order no longer shows in the last bits.
a, _ = planes(core.bm3dvk.BM3Dv2(src, sigma=8, radius=1, extractor_exp=6), 2)
b, _ = planes(core.bm3dvk.BM3Dv2(src, sigma=8, radius=1, extractor_exp=6), 2)
assert np.array_equal(a, b), "extractor_exp output differs between evaluations"
print("extractor determinism: ok")
print("all checks passed")
