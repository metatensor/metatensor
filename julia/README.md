# Metatensor.jl

This package contains the Julia bindings to the C API of metatensor.


## Local development

Getting the code:

```bash
git clone https://github.com/metatensor/metatensor
cd metatensor
```

Building the local version of the core library, re-run this whenever the core
library changes.

```bash
julia julia/deps/build_local.jl
```

To setup the environment, start a Julia REPL by running `julia`, and then

```julia
] add https://github.com/Luthaf/Metatensor_jll.jl
] dev ./julia
```

Finally to develop the code, we recommend using
[Revise](https://timholy.github.io/Revise.jl/) to automatically reload
Metatensor when you make any changes to the code, and working from a Julia REPL.

```julia
# run all the tests
] test Metatensor

# load the package and try the functions yourself
using Metatensor
```
