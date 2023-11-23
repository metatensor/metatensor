# Metatensor.jl

This package contains the Julia bindings to the C API of metatensor.


## Local development

```bash
git clone https://github.com/lab-cosmo/metatensor
cd metatensor
julia
```

From the Julia prompt:

```julia
] add https://github.com/Luthaf/Metatensor_jll.jl
] dev ./julia

# run all the tests
] test Metatensor

# load the package and try the functions
using Metatensor
```

We recommend using [Revise](https://timholy.github.io/Revise.jl/) to
automatically reload Metatensor when you make any changes to the code.
