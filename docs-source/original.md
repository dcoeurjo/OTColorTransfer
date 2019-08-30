# Sliced OT Color Transfer / CPU / Balanced


We discuss some technical details of the Sliced OT implementation in C++. The code use classical STL containers and algorithms,  [stb_image.h/stb_image_writer.h](https://github.com/nothings/stb) for image IO and [Cimg](http://cimg.eu) for the bilateral filter that can be used to regularize the output.

!!! code
    `colorTransfer.cpp`

##Code comments

First, to uniformly sample the set of directions $S^3$, we first create a (seeded) random Number generator following a centered normal distribution law.

``` c++
//Random generator init to draw random line directions
std::mt19937 gen;
gen.seed(10);
std::normal_distribution<float> dist{0.0,1.0};
```


Uniform directions are then selected by normalizing the three realizations of that random number generator[^muller]:



``` c++
//Random direction
float dirx = dist(gen);
float diry = dist(gen);
float dirz = dist(gen);
float norm = sqrt(dirx*dirx + diry*diry + dirz*dirz);
dirx /= norm;
diry /= norm;
dirz /= norm;
```

Then, the core of the method consists in computing the projections:

``` c++
//We project the points
 for(auto i = 0; i < projsource.size(); ++i)
 {
   projsource[i] = dirx * source[3*i] + diry * source[3*i+1] + dirz * source[3*i+2];
   projtarget[i] = dirx * target[3*i] + diry * target[3*i+1] + dirz * target[3*i+2];
 }
```

and sorting the id of the points according to their projection. Sorting the point sets is
the bottleneck of the method. As the two sorts are independent, we can do them in parallel:

``` c++
std::thread threadA([&]{ std::sort(idSource.begin(), idSource.end(), lambdaProjSource); });
std::sort(idTarget.begin(), idTarget.end(), lambdaProjTarget);
threadA.join();
```
 
with the two lambdas

``` c++
 //Lambda expression for the comparison of points in RGB
 //according to their projections
 auto lambdaProjSource = [&projsource](unsigned int a, unsigned int b) {return projsource[a] < projsource[b]; };
 auto lambdaProjTarget = [&projtarget](unsigned int a, unsigned int b) {return projtarget[a] < projtarget[b]; };
```

Then, advection vectors can be accumulated in a batch

``` c++
for(auto i = 0; i < idSource.size(); ++i)
     {
       auto pix = idSource[i];
       advect[3*pix]   += dirx * (projtarget[idTarget[i]] - projsource[idSource[i]]);
       advect[3*pix+1] += diry * (projtarget[idTarget[i]] - projsource[idSource[i]]);
       advect[3*pix+2] += dirz * (projtarget[idTarget[i]] - projsource[idSource[i]]);
     }
```

Before being used to advect the source image points:


``` c++
for(auto i = 0; i <3*N; ++i)
  source[i] += advect[i]/(float)batchSize;
```

## Usage

```
colorTransfer
Usage: ./colorTransfer [OPTIONS]

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

100 slices, default parameters, no regularization (3,5 GHz 6-Core Intel Xeon E5).

```
./colorTransfer -s pexelA-0.png -t pexelB-0.png -o output.png -n 100 --silent
Source image: 1280x1024   (3)
Target image: 1280x1024   (3)
...
elapsed time: 14.5373s
```


[^muller]: Muller, M. E. "A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres." Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.
