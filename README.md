# Deep-Kernel-GP

## Dependencies
The package has numpy and scipy.linalg as dependencies.
The examples also use matplotlib and scikit-learn

## Introduction

## Examples

### Learning a function with varying length scale

In the example.py script, deep kernel learning (DKL) is used to learn from samples of the function sin(64(x+0.5)**4).

Learning this function with a Neural Network would be hard, since it can be challenging to fit rapidly oscilating functions using NNs.
Learning the function using GPRegression with a squared exponential covariance function, would also be suboptimal, since we need to specify a fixed length scale.
Unless we have a lot of samples,we would be forced to give up precision on the slowly varying part of the function.

DKL solves the problem quite nicely:
<p align="center">
  <img src="ex1_1.png" width="350"/>
  <img src="ex1_2.png" width="350"/>
</p>