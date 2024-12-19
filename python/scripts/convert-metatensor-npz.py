#!/usr/bin/env python
import shutil
import sys

import numpy as np


def transform_metatensor_(path):
    data = dict(np.load(path))

    if "blocks/0/values/data" not in data:
        print(f"{path} is already in the new metatensor format")
        return

    cleaned = {}
    for key, value in data.items():
        if key.endswith("/values/data"):
            key = key.replace("/data", "")

        elif key.endswith("/values/samples"):
            key = key.replace("/values/", "/")
        elif "/values/components" in key:
            key = key.replace("/values/", "/")
        elif key.endswith("/properties"):
            key = key.replace("/values/", "/")

        elif "/gradients/" in key and key.endswith("/data"):
            key = key.replace("/data", "/values")

        cleaned[key] = value

    return cleaned


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <file> <file> <file>")

    for path in sys.argv[1:]:
        data = transform_metatensor_(path)
        if data is not None:
            shutil.copyfile(path, path + ".bak")
            np.savez(path, **data)
            print(f"converted {path}, a backup is at {path}.bak")
