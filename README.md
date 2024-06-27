# MultiTaskBayesian

Bayesian optimisation for multiple, parametrised tasks, includes GPU speedups using CUDA

Uses the Eigen linear algebra library, not included since large in size, please extract a release into the lib folder, then compile as follows, (here using version 3.4.0)
```shell
g++ -I ../../lib/eigen-3.4.0/ eigenTest.cpp
```

## Current progress

Currently single task Bayesian optimisation is implemented, with a CPU only and GPU boosted implementation. 

## Overview

A basic CPU only Bayesian optimiser can be found at `src/optimisers/Bayesian.cpp`, with the GPU accelerated version `src/optimisers/BayesianCUDA.cpp`. The surrogate model CUDA kernel is located in `src/optimisers/cuda/SurrogateKernel.cu` - the GPU is used to increase the density of the Monte-Carlo surrogate model evaluations.

Polymorphism is implemented, primarily to simplify development so different optimisers and target merit functions can be combined, an example of this in action is seen in `src/analysis/main.cpp`.
