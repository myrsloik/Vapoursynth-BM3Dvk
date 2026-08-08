import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags


PLUGIN_DIR = Path("vapoursynth") / "plugins" / "bm3dvulkan"


def _library_name() -> str:
    """The plugin filename CMake produces for this platform."""
    if sys.platform == "win32":
        return "bm3dvulkan.dll"
    if sys.platform == "darwin":
        return "libbm3dvulkan.dylib"
    return "libbm3dvulkan.so"


def _built_library(build_dir: Path) -> Path:
    """Where CMake left it. Multi-config generators (MSVC) add a configuration directory,
    single-config ones do not, so both are searched rather than guessed at."""
    candidates = [
        build_dir / "lib" / _library_name(),
        build_dir / "lib" / "Release" / _library_name(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Expected plugin library was not built; looked in: "
        + ", ".join(str(c) for c in candidates)
    )


class CMakeBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return

        # The plugin is a native binary but contains no Python extension module, so the
        # wheel is py3-none-<platform> rather than tied to an interpreter ABI.
        build_data["pure_python"] = False
        build_data["infer_tag"] = False
        if sys.platform == "darwin":
            arch = os.environ.get("CMAKE_OSX_ARCHITECTURES") or os.uname().machine
            if arch not in {"arm64", "x86_64"}:
                raise RuntimeError(f"Unsupported macOS architecture for wheel build: {arch}")
            platform_tag = next(tags.mac_platforms((12, 0), arch))
        else:
            platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        build_data["tag"] = f"py3-none-{platform_tag}"

        root = Path(self.root).resolve()
        build_dir = root / "build" / f"wheel-{platform_tag}"
        output_dir = root / PLUGIN_DIR

        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        configure = [
            "cmake",
            "-S", str(root),
            "-B", str(build_dir),
            "-D", "CMAKE_BUILD_TYPE=Release",
            "-D", f"VAPOURSYNTH_INCLUDE_DIRECTORY={self._vapoursynth_include_dir()}",
        ]
        if sys.platform == "darwin":
            env["MACOSX_DEPLOYMENT_TARGET"] = "12.0"
            configure += [
                "-D", "CMAKE_OSX_DEPLOYMENT_TARGET=12.0",
                "-D", f"CMAKE_OSX_ARCHITECTURES={arch}",
            ]
        subprocess.run(configure, check=True, env=env)
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release"], check=True, env=env
        )

        shutil.copy2(_built_library(build_dir), output_dir / _library_name())
        # The manifest names the library without its extension, which the core appends --
        # but it keeps any "lib" prefix, so the entry differs per platform and has to be
        # written from the real filename rather than shipped as a fixed file.
        (output_dir / "manifest.vs").write_text(
            "[VapourSynth Manifest V1]\n" + Path(_library_name()).stem + "\n",
            encoding="utf-8",
        )
        build_data["force_include"][str(output_dir / _library_name())] = str(
            PLUGIN_DIR / _library_name()
        )
        build_data["force_include"][str(output_dir / "manifest.vs")] = str(
            PLUGIN_DIR / "manifest.vs"
        )

    def clean(self, versions):
        root = Path(self.root).resolve()
        shutil.rmtree(root / PLUGIN_DIR, ignore_errors=True)

    @staticmethod
    def _vapoursynth_include_dir():
        command = [
            sys.executable,
            "-c",
            "import vapoursynth; print(vapoursynth.get_include(), end='')",
        ]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        include_dir = Path(completed.stdout)
        if not include_dir.is_dir():
            raise RuntimeError(f"VapourSynth include directory does not exist: {include_dir}")
        return include_dir
