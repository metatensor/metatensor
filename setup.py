import os
import sys
import subprocess

from setuptools import setup, Extension
from wheel.bdist_wheel import bdist_wheel
from distutils.command.build_ext import build_ext  # type: ignore

ROOT = os.path.realpath(os.path.dirname(__file__))

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")


EQUISTORE_BUILD_TYPE = os.environ.get("EQUISTORE_BUILD_TYPE", "release")
if EQUISTORE_BUILD_TYPE not in ["debug", "release"]:
    raise Exception(
        f"invalid build type passed: '{EQUISTORE_BUILD_TYPE}', "
        "expected 'debug' or 'release'"
    )


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3.6. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class cmake_ext(build_ext):
    """
    Build the native library using cmake
    """

    def run(self):
        source_dir = ROOT
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(os.path.realpath(self.build_lib), "equistore")

        try:
            os.mkdir(build_dir)
        except OSError:
            pass

        cmake_options = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DCMAKE_BUILD_TYPE={EQUISTORE_BUILD_TYPE}",
            "-DBUILD_SHARED_LIBS=ON",
            "-DEQUISTORE_BUILD_FOR_PYTHON=ON",
        ]

        if "CARGO" in os.environ:
            cmake_options.append(f"-DCARGO_EXE={os.environ['CARGO']}")

        subprocess.run(
            ["cmake", source_dir, *cmake_options],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "--target", "install"],
            check=True,
        )


# read version from Cargo.toml
with open("Cargo.toml") as fd:
    for line in fd:
        if line.startswith("version"):
            _, version = line.split(" = ")
            # remove quotes
            version = version[1:-2]
            # take the first version in the file, this should be the right
            # version
            break

setup(
    version=version,
    ext_modules=[
        # only declare the extension, it is built & copied as required by cmake
        # in the build_ext command
        Extension(name="equistore", sources=[]),
    ],
    cmdclass={
        "build_ext": cmake_ext,
        "bdist_wheel": universal_wheel,
    },
    package_data={
        "equistore": [
            "equistore/lib/*",
            "equistore/include/*",
        ]
    },
)
