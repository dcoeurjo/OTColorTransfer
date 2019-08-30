# Sliced OT Color Transfer / CPU / Balanced


We discuss some technical details of the Sliced OT implementation in C++. The code use classical STL containers and algorithm,  [stb_image.h/stb_image_writer.h](https://github.com/nothings/stb) for image IO and [Cimg](http://cimg.eu) for the bilateral filter that can be used to regularize the output.

The Sliced Partial Optimal Transport is computed using the optimized [SPOT](https://github.com/nbonneel/spot/) code ([Nicolas Bonneel](https://perso.liris.cnrs.fr/nicolas.bonneel/), [David Coeurjolly](https://perso.liris.cnrs.fr/david.coeurjolly/)).

!!! code
    `colorTransferPartial.cpp`

##Code comments

## Usage

```
colorTransferPartial
Usage: ./colorTransferPartial [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -s,--source TEXT            Source image
  -t,--target TEXT            Target image
  -o,--output TEXT            Output image
  -n,--nbsteps UINT           Number of sliced steps (3)
  -b,--sizeBatch UINT         Number of dirtections on a batch (1)
  -r,--regularization         Apply a regularization step of the transport plan using bilateral filter (false).
  --sigmaXY FLOAT             Sigma parameter in the spatial domain for the bilateral regularization (16.0)
  --sigmaV FLOAT              Sigma parameter in the value domain for the bilateral regularization (5.0)
  --silent                    No verbose messages
```

## Timings

100 slices, default parameters, no regularization, same image size (3,5 GHz 6-Core Intel Xeon E5). When considering images with same size, the code has an overhead compared to the CPU/Balanced code.

```
./colorTransferPartial -s pexelA-0.png -t pexelB-0.png -o output.png -n 100 --silent
Source image: 1280x1024   (3)
Target image: 1280x1024   (3)
...
elapsed time: 34.1797s
```

When we increase the size of the target image to reproduce the results of [Sliced Partial Optimal Transport](https://perso.liris.cnrs.fr/nicolas.bonneel/spot/):
```
./colorTransferPartial -s pexelA-0.png -t pexelB-0-larger.png -o output-partial.png -n 100
Source image: 1280x1024   (3)
Target image: 1536x1229   (3)
finished computation at Fri Aug 30 13:25:11 2019
elapsed time: 33.4571s
Exporting..
```
