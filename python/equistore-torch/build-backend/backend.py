# this is a custom Python build backend wrapping setuptool's to add a build-time
# dependencies to equistore-core, using the local version if it exists, and
# otherwise falling back to the one on PyPI.
import os
import uuid

from setuptools import build_meta


ROOT = os.path.realpath(os.path.dirname(__file__))
EQUISTORE_CORE = os.path.realpath(os.path.join(ROOT, "..", "..", "equistore-core"))
if os.path.exists(EQUISTORE_CORE):
    # we are building from a git checkout

    # add a random uuid to the file url to prevent pip from using a cached
    # wheel for equistore-core, and force it to re-build from scratch
    uuid = uuid.uuid4()
    EQUISTORE_CORE_DEP = f"equistore-core @ file://{EQUISTORE_CORE}?{uuid}"
else:
    # we are building from a sdist
    EQUISTORE_CORE_DEP = "equistore-core >=0.1.0.dev0,<0.2.0"


prepare_metadata_for_build_wheel = build_meta.prepare_metadata_for_build_wheel
build_wheel = build_meta.build_wheel
build_sdist = build_meta.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    defaults = build_meta.get_requires_for_build_wheel(config_settings)
    return defaults + [EQUISTORE_CORE_DEP]


def get_requires_for_build_sdist(config_settings=None):
    defaults = build_meta.get_requires_for_build_sdist(config_settings)
    return defaults + [EQUISTORE_CORE_DEP]
