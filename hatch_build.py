import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags


MACOS_MIN_VERSION = (12, 0)
PLUGIN_LIBRARY = "libbm3dmetal.dylib"
PLUGIN_DIR = Path("vapoursynth") / "plugins" / "bm3dmetal"


class CMakeBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return

        if sys.platform != "darwin":
            raise RuntimeError("vapoursynth-bm3dmetal wheels can only be built on macOS.")

        arch = os.environ.get("CMAKE_OSX_ARCHITECTURES") or platform.machine()
        if arch not in {"arm64", "x86_64"}:
            raise RuntimeError(f"Unsupported macOS architecture for wheel build: {arch}")

        platform_tag = next(tags.mac_platforms(MACOS_MIN_VERSION, arch))
        build_data["pure_python"] = False
        build_data["infer_tag"] = False
        build_data["tag"] = f"py3-none-{platform_tag}"

        root = Path(self.root).resolve()
        build_dir = root / "build" / f"wheel-{arch}"
        output_dir = root / PLUGIN_DIR
        staging_dylib = output_dir / PLUGIN_LIBRARY

        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        include_dir = self._vapoursynth_include_dir()
        env = os.environ.copy()
        env["MACOSX_DEPLOYMENT_TARGET"] = ".".join(map(str, MACOS_MIN_VERSION))

        configure = [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build_dir),
            "-D",
            "CMAKE_BUILD_TYPE=Release",
            "-D",
            f"CMAKE_OSX_DEPLOYMENT_TARGET={env['MACOSX_DEPLOYMENT_TARGET']}",
            "-D",
            f"CMAKE_OSX_ARCHITECTURES={arch}",
            "-D",
            f"VAPOURSYNTH_INCLUDE_DIRECTORY={include_dir}",
        ]
        subprocess.run(configure, check=True, env=env)
        subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release"], check=True, env=env)

        built_dylib = build_dir / "lib" / PLUGIN_LIBRARY
        if not built_dylib.is_file():
            raise RuntimeError(f"Expected plugin library was not built: {built_dylib}")

        shutil.copy2(built_dylib, staging_dylib)
        shutil.copy2(root / "packaging" / "manifest.vs", output_dir / "manifest.vs")

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
