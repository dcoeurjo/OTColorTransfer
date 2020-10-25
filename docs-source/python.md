# Sliced OT Color Transfer / Python exampe

Here you have a [python implementation](https://github.com/dcoeurjo/OTColorTransfer/issues/2) by
[@iperov](https://github.com/iperov) (thanks!). This code also illustrates the
sliced OT color transfer in Lab color space. This code requires numpy
and opencv python packages.


``` python
import numpy as np
from numpy import linalg as npla
import cv2
def CTSOT(src,trg, steps=10, batch_size=5, reg_sigmaXY=16.0, reg_sigmaV=5.0):
    """
    Color Transform via Sliced Optimal Transfer, ported by @iperov

    src         - any float range any channel image
    dst         - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter
    
    return value - clip it manually
    """
    if not np.issubdtype(src.dtype, np.floating):
        raise ValueError("src value must be float")
    if not np.issubdtype(trg.dtype, np.floating):
        raise ValueError("trg value must be float")

    if len(src.shape) != 3:
        raise ValueError("src shape must have rank 3 (h,w,c)")
    
    if src.shape != trg.shape:
        raise ValueError("src and trg shapes must be equal")    

    src_dtype = src.dtype        
    h,w,c = src.shape
    new_src = src.copy()

    for step in range (steps):
        advect = np.zeros ( (h*w,c), dtype=src_dtype )
        for batch in range (batch_size):
            dir = np.random.normal(size=c).astype(src_dtype)
            dir /= npla.norm(dir)

            projsource = np.sum( new_src*dir, axis=-1).reshape ((h*w))
            projtarget = np.sum( trg*dir, axis=-1).reshape ((h*w))

            idSource = np.argsort (projsource)
            idTarget = np.argsort (projtarget)

            a = projtarget[idTarget]-projsource[idSource]
            for i_c in range(c):
                advect[idSource,i_c] += a * dir[i_c]
        new_src += advect.reshape( (h,w,c) ) / batch_size

    if reg_sigmaXY != 0.0:
        src_diff = new_src-src
        new_src = src + cv2.bilateralFilter (src_diff, 0, reg_sigmaV, reg_sigmaXY )
    return new_src

``` 
