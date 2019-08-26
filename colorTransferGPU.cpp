#include <iostream>
#include <string>
#include <random>
#include <vector>

#include "CLI11.hpp"

#define cimg_display 0
#include "CImg.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>
#include <boost/compute/algorithm/count_if.hpp>
#include <boost/compute/function.hpp>
#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/algorithm/sort_by_key.hpp>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/algorithm/scatter.hpp>
#include <boost/compute/algorithm/accumulate.hpp>

#include <boost/range/adaptor/strided.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/assign.hpp>
#include <boost/range/algorithm.hpp>


//Global flag to silent verbose messages
bool silent;

namespace compute = boost::compute;

void slicedTransfer(std::vector<float> &source,
const std::vector<float> &target,
const int nbSteps,
const int batchSize)
{
  
  //Random generator init to draw random line directions
  std::mt19937 gen(0);
  std::normal_distribution<float> dist{0.0,1.0};
  
  const auto N = source.size() / 3;
  
  // get default OpenCL device and setup context
  compute::device device = compute::system::default_device();
  compute::context context(device);
  compute::command_queue queue(context, device);
  
  //Sending buffers
  compute::vector<float> sourceGPUx(N, context);
  compute::vector<float> sourceGPUy(N, context);
  compute::vector<float> sourceGPUz(N, context);
  compute::vector<float> targetGPUx(N, context);
  compute::vector<float> targetGPUy(N, context);
  compute::vector<float> targetGPUz(N, context);
  
  std::vector<float> tmp(N);
  for(auto i=0; i< N; ++i) tmp[i] = source[3*i];
  compute::copy(tmp.begin(), tmp.end(), sourceGPUx.begin(), queue);
  
  for(auto i=0; i< N; ++i) tmp[i] = source[3*i+1];
  compute::copy(tmp.begin(), tmp.end(), sourceGPUy.begin(), queue);
  
  for(auto i=0; i< N; ++i) tmp[i] = source[3*i+2];
  compute::copy(tmp.begin(), tmp.end(), sourceGPUz.begin(), queue);
  
  for(auto i=0; i< N; ++i) tmp[i] = target[3*i];
  compute::copy(tmp.begin(), tmp.end(), targetGPUx.begin(), queue);
  
  for(auto i=0; i< N; ++i) tmp[i] = target[3*i+1];
  compute::copy(tmp.begin(), tmp.end(), targetGPUy.begin(), queue);
  
  for(auto i=0; i< N; ++i) tmp[i] = target[3*i+2];
  compute::copy(tmp.begin(), tmp.end(), targetGPUz.begin(), queue);
  
  
  //Advection vector
  compute::vector<float> advectX(N,context);
  compute::vector<float> advectY(N,context);
  compute::vector<float> advectZ(N,context);
  
  //To store the 1D projections
  compute::vector<float> projsource(N, context);
  compute::vector<float> projtarget(N, context);
  
  compute::vector<float> delta(N, context);
  
  //Pixel Id
  compute::vector<unsigned int> idSource(N, context);
  compute::vector<unsigned int> idTarget(N, context);
  
  
  if (!silent)
  {
    std::cout<<" TSource points: "<<std::endl;
    compute::copy(sourceGPUx.begin(), sourceGPUx.end(), std::ostream_iterator<float>(std::cout, " "), queue);
    std::cout<<std::endl;
    compute::copy(sourceGPUy.begin(), sourceGPUy.end(), std::ostream_iterator<float>(std::cout, " "), queue);
    std::cout<<std::endl;
    compute::copy(sourceGPUz.begin(), sourceGPUz.end(), std::ostream_iterator<float>(std::cout, " "), queue);
    std::cout<<std::endl;
    std::cout<<std::endl;
  }
  if (!silent)
  {
    std::cout<<" Target points: "<<std::endl;
    compute::copy(targetGPUx.begin(), targetGPUx.end(), std::ostream_iterator<float>(std::cout, " "), queue);
    std::cout<<std::endl;
    compute::copy(targetGPUy.begin(), targetGPUy.end(), std::ostream_iterator<float>(std::cout, " "), queue);
    std::cout<<std::endl;
    compute::copy(targetGPUz.begin(), targetGPUz.end(), std::ostream_iterator<float>(std::cout, " "), queue);
    std::cout<<std::endl;
    std::cout<<std::endl;
  }
  
  using boost::compute::lambda::_1;
  using boost::compute::lambda::_2;
  
  for(auto step = 0 ; step < nbSteps; ++step)
  {
    compute::fill(advectX.begin(), advectX.end(), 0.0, queue);
    compute::fill(advectY.begin(), advectY.end(), 0.0, queue);
    compute::fill(advectZ.begin(), advectZ.end(), 0.0, queue);
    
    queue.finish();
    
    for(auto batch = 0; batch < batchSize; ++batch )
    {
      queue.finish();
      //Random direction
      float dirx = dist(gen);
      float diry = dist(gen);
      float dirz = dist(gen);
      float norm = sqrt(dirx*dirx + diry*diry + dirz*dirz);
      dirx /= norm;
      diry /= norm;
      dirz /= norm;
      if (step==1)
      {
        dirx = 1.0;///= norm;
        diry = 0; //= norm;
        dirz = 0.0; ///= norm;
      }
      if (step==0)
      {
        dirx = 0.0;///= norm;
        diry = 1.0; //= norm;
        dirz = 0.0; ///= norm;
        
      }
      if (step==2)
      {
        dirx = 0.0;///= norm;
        diry = 0.0; //= norm;
        dirz = 1.0; ///= norm;
        
      }
      //if (!silent)
      std::cout<<"Slice "<<step<<" batch "<<batch<<"  ("<<dirx<<","<<diry<<","<<dirz<<")"<<std::endl;
      
      //We project the points
      //compute::fill(projsource.begin(), projsource.end(), 0.0, queue);
      // compute::fill(projtarget.begin(), projtarget.end(), 0.0, queue);
      queue.finish();
      
      compute::transform(sourceGPUx.begin(), sourceGPUx.end(), sourceGPUy.begin(), projsource.begin(), _1*dirx+_2*diry,  queue);
      queue.finish();
      compute::transform(sourceGPUz.begin(), sourceGPUz.end(), projsource.begin(), projsource.begin(), _1*dirz + _2,  queue);
      queue.finish();
      
      // projTarget
      compute::transform(targetGPUx.begin(), targetGPUx.end(), targetGPUy.begin(), projtarget.begin(),  _1*dirx + _2*diry,  queue);
      queue.finish();
      compute::transform(targetGPUz.begin(), targetGPUz.end(), projtarget.begin(), projtarget.begin(),  _1*dirz + _2,  queue);
      
      if (!silent)
      {
        std::cout<<"Avant tri"<<std::endl;
        compute::copy(projsource.begin(), projsource.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        std::cout<<"Id: ";
        compute::copy(idSource.begin(), idSource.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        compute::copy(projtarget.begin(), projtarget.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        std::cout<<"Id: ";
        compute::copy(idTarget.begin(), idTarget.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        std::cout<<std::endl;
      }
      queue.finish();
      
      compute::iota(idSource.begin(), idSource.end(), 0, queue);
      compute::iota(idTarget.begin(), idTarget.end(), 0, queue);
      
      queue.finish();
      //1D optimal transport of the projections with two sorts
      compute::sort_by_key(projsource.begin(),projsource.end(), idSource.begin(), queue);
      compute::sort_by_key(projtarget.begin(),projtarget.end(), idTarget.begin(), queue);
      queue.finish();
      
      if (!silent)
      {std::cout<<"Apres tri"<<std::endl;
        compute::copy(projsource.begin(), projsource.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        std::cout<<"Id: ";
        compute::copy(idSource.begin(), idSource.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        compute::copy(projtarget.begin(), projtarget.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        std::cout<<"Id: ";
        compute::copy(idTarget.begin(), idTarget.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        std::cout<<std::endl;
        std::cout<<std::endl;
      }
      
      queue.finish();
      
      compute::transform(projtarget.begin(), projtarget.end(), projsource.begin(),  delta.begin(), compute::minus<float>(), queue);
      //delta[i] is the shift for the i-th point along the projection (idSource[i] point in Source)
      queue.finish();
      
      //Reorder delta[i]
      if (!silent)
      {
        std::cout<<"scatter"<<std::endl;
        compute::copy(delta.begin(), delta.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        std::cout<<"Id: ";compute::copy(idSource.begin(), idSource.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        
      }
      
      compute::scatter(delta.begin(), delta.end(), idSource.begin(), delta.begin(), queue);
      if (!silent)
      {
        std::cout<<"scatter res(source) ";
        compute::copy(delta.begin(), delta.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
      }
      queue.finish();
      
      compute::transform(advectX.begin(), advectX.end(), delta.begin(), advectX.begin(), _1 + _2*dirx, queue);
      queue.finish();
      compute::transform(advectY.begin(), advectY.end(), delta.begin(), advectY.begin(), _1 + _2*diry, queue);
      queue.finish();
      compute::transform(advectZ.begin(), advectZ.end(), delta.begin(), advectZ.begin(), _1 + _2*dirz, queue);
      queue.finish();
      
      if (!silent)
      {
        std::cout<<" Advect vectors: "<<std::endl;
        compute::copy(advectX.begin(), advectX.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        compute::copy(advectY.begin(), advectY.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        compute::copy(advectZ.begin(), advectZ.end(), std::ostream_iterator<float>(std::cout, " "), queue);
        std::cout<<std::endl;
        std::cout<<std::endl;
      }
      
      queue.finish();
      
    }
    //Advection
    compute::transform(sourceGPUx.begin(), sourceGPUx.end(),advectX.begin(),sourceGPUx.begin(), _1 + _2/(float)batchSize, queue);
    queue.finish();
    compute::transform(sourceGPUy.begin(), sourceGPUy.end(),advectY.begin(),sourceGPUy.begin(), _1 + _2/(float)batchSize, queue);
    queue.finish();
    compute::transform(sourceGPUz.begin(), sourceGPUz.end(),advectZ.begin(),sourceGPUz.begin(), _1 + _2/(float)batchSize, queue);
    queue.finish();
    
    if (!silent)
    {
      std::cout<<" Source advected points: "<<std::endl;
      compute::copy(sourceGPUx.begin(), sourceGPUx.end(), std::ostream_iterator<float>(std::cout, " "), queue);
      std::cout<<std::endl;
      compute::copy(sourceGPUy.begin(), sourceGPUy.end(), std::ostream_iterator<float>(std::cout, " "), queue);
      std::cout<<std::endl;
      compute::copy(sourceGPUz.begin(), sourceGPUz.end(), std::ostream_iterator<float>(std::cout, " "), queue);
      std::cout<<std::endl;
      std::cout<<std::endl;
    }
    
  }
  
  //Final copy to the CPU
  std::vector<float> sx(N);
  std::vector<float> sy(N);
  std::vector<float> sz(N);
  compute::copy(sourceGPUx.begin(), sourceGPUx.end(), sx.begin(), queue);
  compute::copy(sourceGPUy.begin(), sourceGPUy.end(), sy.begin(), queue);
  compute::copy(sourceGPUz.begin(), sourceGPUz.end(), sz.begin(), queue);
  for(auto i=0; i < N; ++i)
  {
    source[3*i]   = sx[i];
    source[3*i+1] = sy[i];
    source[3*i+2] = sz[i];
  }
}

int main(int argc, char **argv)
{
  CLI::App app{"colorTransfer"};
  std::string sourceImage="pexelAred.png";
  app.add_option("-s,--source", sourceImage, "Source image");
  std::string targetImage="pexelBred.png";
  app.add_option("-t,--target", targetImage, "Target image");
  std::string outputImage= "output.png";
  app.add_option("-o,--output", outputImage, "Output image");
  unsigned int nbSteps = 3;
  app.add_option("-n,--nbsteps", nbSteps, "Number of sliced steps (3)");
  unsigned int batchSize = 1;
  app.add_option("-b,--sizeBatch", batchSize, "Number of dirtections on a batch (1)");
  bool applyRegularization = false;
  app.add_flag("-r,--regularization", applyRegularization, "Apply a regularization step of the transport plan using bilateral filter (false).");
  float sigmaXY = 16.0;
  app.add_option("--sigmaXY", sigmaXY, "Sigma parameter in the spatial domain for the bilateral regularization (16.0)");
  float sigmaV = 5.0;
  app.add_option("--sigmaV", sigmaV, "Sigma parameter in the value domain for the bilateral regularization (5.0)");
  silent = false;
  app.add_flag("--silent", silent, "No verbose messages");
  CLI11_PARSE(app, argc, argv);
  
  //Image loading
  int width,height, nbChannels;
  unsigned char *source = stbi_load(sourceImage.c_str(), &width, &height, &nbChannels, 0);
  if (!silent) std::cout<< "Source image: "<<width<<"x"<<height<<"   ("<<nbChannels<<")"<< std::endl;
  int width_target,height_target, nbChannels_target;
  unsigned char *target = stbi_load(targetImage.c_str(), &width_target, &height_target, &nbChannels_target, 0);
  if (!silent) std::cout<< "Target image: "<<width_target<<"x"<<height_target<<"   ("<<nbChannels_target<<")"<< std::endl;
  
  if ((width != width_target) || (height_target != height) || (nbChannels!=nbChannels_target))
  {
    std::cout<< "Image sizes do not match. "<<std::endl;
    exit(1);
  }
  if (nbChannels <3)
  {
    std::cout<< "Input images must be color images."<<std::endl;
    exit(1);
  }
  
  std::vector<float> sourcefloat(width*height*nbChannels);
  std::vector<float> targetfloat(width*height*nbChannels);
#pragma omp parallel for
  for(auto i = 0 ; i <width*height*nbChannels; ++i)
  {
    sourcefloat[i] = static_cast<float>(source[i]);
    targetfloat[i] = static_cast<float>(target[i]);
  }
  
  //Main computation
  slicedTransfer(sourcefloat, targetfloat, nbSteps, batchSize);
  
  
  //Output
  std::vector<unsigned char> output(width*height*nbChannels);
  if (applyRegularization)
  {
    //Regularization of the transport plan (optional)
    // (bilateral filter of the difference)
    if (!silent) std::cout<<"Applying regularization step"<<std::endl;
    cimg_library::CImg<float> transport(width, height, 1, 3);
    for(auto i=0; i<width*height; ++i)
    {
      transport[i] = sourcefloat[3*i] - static_cast<float>(source[3*i]);
      transport[i+ width*height] = sourcefloat[3*i+1] - static_cast<float>(source[3*i+1]);
      transport[i+2*width*height] = sourcefloat[3*i+2] - static_cast<float>(source[3*i+2]);
    }
    transport.blur_bilateral(transport, sigmaXY,sigmaV);
    
    auto output2(output);
    for(auto i = 0 ; i < width*height ; ++i)
    {
      output[3*i]   = static_cast<unsigned char>(  std::min(255.0f, std::max(0.0f, static_cast<float>(source[3*i  ]) + transport[i])));
      output[3*i+1] = static_cast<unsigned char>(  std::min(255.0f, std::max(0.0f, static_cast<float>(source[3*i+1]) + transport[i+ width*height])));
      output[3*i+2] = static_cast<unsigned char>(  std::min(255.0f, std::max(0.0f, static_cast<float>(source[3*i+2]) + transport[i+ width*height*2])));
    }
  }
  else
  {
    for(auto i = 0 ; i < width*height*nbChannels ; ++i)
      output[i] = static_cast<unsigned char>(  std::min(255.0f, std::max(0.0f,  sourcefloat[i])));
  }
  
  //Final export
  if (!silent) std::cout<<"Exporting.."<<std::endl;
  int errcode = stbi_write_png(outputImage.c_str(), width, height, nbChannels, output.data(), nbChannels*width);
  if (!errcode)
  {
    std::cout<<"Error while exporting the resulting image."<<std::endl;
    exit(errcode);
  }
  
  stbi_image_free(source);
  stbi_image_free(target);
  exit(0);
}
