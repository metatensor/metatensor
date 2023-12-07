import sys

import torch  # noqa


PYTEST_DONT_REWRITE = '"""PYTEST_DONT_REWRITE"""'

if __name__ == "__main__":
    try:
        # `torch.autograd.gradcheck` is the `torch.autograd.gradcheck.gradcheck``
        # function, so we need to pick the module from `sys.modules`
        path = sys.modules["torch.autograd.gradcheck"].__file__

        with open(path) as fd:
            content = fd.read()

        if PYTEST_DONT_REWRITE in content:
            sys.exit(0)

        with open(path, "w") as fd:
            print(f"rewriting {path} to add PYTEST_DONT_REWRITE")
            fd.write(PYTEST_DONT_REWRITE)
            fd.write("\n")
            fd.write(content)

    except Exception:
        print(
            "failed to add PYTEST_DONT_REWRITE to `torch.autograd.gradcheck`,",
            "tests are likely to fail",
            file=sys.stderr,
        )
