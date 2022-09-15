Overview
========

What is Equistore?
--------
Storage format for the specific kind of data one comes across in atomistic ML 

Equistore at a glance
--------
What kinds of data structures you might encounter

Subsubheading  (what it is)
#######
TensorMap 
Tensor Blocks
Labels

Gradients and how we manage them 
-------
Gradient samples - "special" format 

    first sample of gradients is "sample" that refers to the row in block.values that we are taking the gradient of. 
the other samples - what we are taking the gradient with respect to. 
Write what this entails -- block.gradients.sample (i A j) (pair feature i j A k)

Cell gradients  - Sample (i) 
components [[x y z ] [x y z]] (displacement matrix) 

Gradient wrt hypers








