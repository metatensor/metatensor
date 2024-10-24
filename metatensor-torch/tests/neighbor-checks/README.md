# Tests for neighbors list implementations

This directory contains data to help tests neighbor list calculation when
interfacing a new engine with metatensor atomistic models.

Each JSON file contains a system, some `NeighborListOptions` and the expected
set of pairs that should be produced (the pairs can be in a different order).
For a half neighbor list, the pairs are also permitted to be in reverse order:

- sample `[i, j, A, B, C]` becomes `[j, i, -A, -B, -C]`;
- distance `[x, y, z]` becomes `[-x, -y, -z]`.


When writing a new interface between metatensor and a simulation engine, you can
use these files to ensure your neighbor list implementation follows the expected
behavior for metatensor models.

If your engine can produce non-`strict` neighbor lists, you can still use these
files for testing. The pairs in the test files then correspond to the minimal
set that must be included in the output, and any additional pair must be above
the cutoff.
