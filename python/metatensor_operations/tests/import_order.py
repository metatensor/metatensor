# Make sure operations are properly re-exported when metatensor_operations is imported
# before metatensor.
import subprocess
import sys


def test_import_order():
    # run in a subprocess since the current one may have already loaded the modules

    code = """
import metatensor_operations
import metatensor as mts

assert hasattr(mts, "allclose")
assert hasattr(mts, "allclose_block")
assert hasattr(mts, "block_from_array")
assert hasattr(mts, "join")
assert hasattr(mts, "split")
assert hasattr(mts, "zeros_like")
"""

    subprocess.run([sys.executable, "-c", code], check=True)
