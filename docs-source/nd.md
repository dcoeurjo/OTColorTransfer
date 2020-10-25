# nD Sliced Optimal Transport

As the sliced formulation scales linearly with the dimension, the nD version of the
transfer is straightforward.

!!! code
    `ndTransfer.cpp`


The I/O is an ASCII file with the following structure (one line per "pixel", scalars in column, same number of scalars, aka same dimension, for all points):

```
i j a b c d e ...
....
```

with $i,j\in\mathbb{Z}$ and the remaining values are in $[0,1)^d$. The $(i,j)$ values
are only used for the per channel bilateral filter to regularize the transport plan.
