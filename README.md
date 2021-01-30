# nlm-jax

Vectorized and differentiable implementation of non-local means by [Buades et. al](https://ieeexplore.ieee.org/document/1467423) for image denoising in JAX for EE367.

Approximately two orders of magnitude speed up with JIT compilation + GPU acceleration compared to a raw NumPy implementation.
