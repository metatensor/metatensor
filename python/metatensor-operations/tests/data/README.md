# Tests data files for metatensor-models

- **qm7-spherical-expansion**: SOAP spherical expansion for the first 10
  structures in QM7, computed with rascaline (commit 81218aca), with 'cell'
  and 'positions' gradients, using the following hyper-parameters:

```py
cutoff=3.5,
max_radial=4,
max_angular=4,
atomic_gaussian_width=0.3,
radial_basis={"Gto": {}},
center_atom_weight=1.0,
cutoff_function={"ShiftedCosine": {"width": 0.5}},
```



- **qm7-power-spectrum**: SOAP power spectrum for the first 10 structures in
  QM7, computed with rascaline (commit 81218aca), with 'cell' and 'positions'
  gradients, using the following hyper-parameters:

```py
cutoff=3.5,
max_radial=4,
max_angular=4,
atomic_gaussian_width=0.3,
radial_basis={"Gto": {}},
center_atom_weight=1.0,
cutoff_function={"ShiftedCosine": {"width": 0.5}},
```
