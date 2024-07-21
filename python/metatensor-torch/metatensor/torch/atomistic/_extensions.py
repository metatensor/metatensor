import hashlib
import os
import shutil
import site

import torch

from .. import _c_lib


METATENSOR_TORCH_LIB_PATH = _c_lib._lib_path()


def _rascaline_lib_path():
    import rascaline

    return [rascaline._c_lib._lib_path()]


# Manual definition of which TorchScript extensions have their own dependencies. The
# dependencies should be returned in the order they need to be loaded.
EXTENSIONS_WITH_DEPENDENCIES = {
    "rascaline_torch": _rascaline_lib_path,
}


def _collect_extensions(extensions_path):
    """
    Record the list of loaded TorchScript extensions (and their dependencies), to check
    that they are also loaded when executing the model.
    """
    if extensions_path is not None:
        if os.path.exists(extensions_path):
            shutil.rmtree(extensions_path)
        os.makedirs(extensions_path)
        # TODO: the extensions are currently collected in a separate directory,
        # should we store the files directly inside the model file? This would makes
        # the model platform-specific but much more convenient (since the end user
        # does not have to move a model around)

    extensions = []
    extensions_deps = []
    for library in torch.ops.loaded_libraries:
        if library == METATENSOR_TORCH_LIB_PATH:
            continue

        path = _copy_extension(library, extensions_path)
        info = _extension_info(library)
        info["path"] = path
        extensions.append(info)

        for extra in EXTENSIONS_WITH_DEPENDENCIES.get(info["name"], lambda: [])():
            path = _copy_extension(extra, extensions_path)
            info = _extension_info(extra)
            info["path"] = path
            extensions_deps.append(info)

    return extensions, extensions_deps


def _copy_extension(full_path, extensions_path):
    site_packages = site.getsitepackages()
    if site.ENABLE_USER_SITE:
        site_packages.append(site.getusersitepackages())

    path = full_path
    for prefix in site_packages:
        # Remove any site-package prefix
        if path.startswith(prefix):
            path = os.path.relpath(path, prefix)
            break

    if extensions_path is not None:
        collect_path = os.path.join(extensions_path, path)
        if os.path.exists(collect_path):
            raise RuntimeError(
                f"more than one extension would be collected at {collect_path}"
            )

        os.makedirs(os.path.dirname(collect_path), exist_ok=True)
        shutil.copyfile(full_path, collect_path)

    return path


def _extension_info(path):
    # get the name of the library, excluding any shared object prefix/suffix
    name = os.path.basename(path)
    if name.startswith("lib"):
        name = name[3:]

    if name.endswith(".so"):
        name = name[:-3]

    if name.endswith(".dll"):
        name = name[:-4]

    if name.endswith(".dylib"):
        name = name[:-6]

    # Collect the hash of the extension shared library. We don't currently use
    # this, but it would allow for binary-level reproducibility later.
    with open(path, "rb") as fd:
        sha256 = hashlib.sha256(fd.read()).hexdigest()

    return {"name": name, "sha256": sha256}
