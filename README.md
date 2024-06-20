# MultiTaskBayesian

Bayesian optimisation for multiple, parametrised tasks, includes GPU speedups using CUDA

Uses the Eigen linear algebra library, not included since large in size, please extract a release into the lib folder, then compile as follows, (here using version 3.4.0)
```shell
g++ -I ../../lib/eigen-3.4.0/ eigenTest.cpp
```

## Current progress

Currently single task Bayesian optimisation is implemented, with the CUDA code for inner loop optimisations being developed.
