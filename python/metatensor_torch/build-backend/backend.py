# This is a custom Python build backend wrapping setuptool's to add a build-time
# dependencies to metatensor-core, using the local version if it exists, and otherwise
# falling back to the one on PyPI.
#
# This also allows to only depend on torch/cmake when building the wheel and not the
# sdist
import os

from setuptools import build_meta


ROOT = os.path.realpath(os.path.dirname(__file__))
METATENSOR_CORE = os.path.realpath(os.path.join(ROOT, "..", "..", "metatensor_core"))
FORCED_METATENSOR_CORE_VERSION = os.environ.get(
    "METATENSOR_TORCH_BUILD_WITH_METATENSOR_CORE_VERSION"
)
METATENSOR_NO_LOCAL_DEPS = os.environ.get("METATENSOR_NO_LOCAL_DEPS", "0") == "1"


if FORCED_METATENSOR_CORE_VERSION is not None:
    # force a specific version for metatensor-core, this is used when checking the build
    # from a sdist on a non-released version
    METATENSOR_CORE_DEP = f"metatensor-core =={FORCED_METATENSOR_CORE_VERSION}"

elif not METATENSOR_NO_LOCAL_DEPS and os.path.exists(METATENSOR_CORE):
    # we are building from a git checkout
    METATENSOR_CORE_DEP = f"metatensor-core @ file://{METATENSOR_CORE}"
else:
    # we are building from a sdist
    METATENSOR_CORE_DEP = "metatensor-core >=0.1.0,<0.2.0"


FORCED_TORCH_VERSION = os.environ.get("METATENSOR_TORCH_BUILD_WITH_TORCH_VERSION")
if FORCED_TORCH_VERSION is not None:
    TORCH_DEP = f"torch =={FORCED_TORCH_VERSION}"
else:
    TORCH_DEP = "torch >=1.12"

# ==================================================================================== #
#                   Build backend functions definition                                 #
# ==================================================================================== #

# Use the default version of these
prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
get_requires_for_build_sdist = build_meta.get_requires_for_build_sdist
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


# Special dependencies to build the wheels
def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + [TORCH_DEP, METATENSOR_CORE_DEP]
