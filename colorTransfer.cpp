/*
 Copyright (c) 2019 CNRS
 David Coeurjolly <david.coeurjolly@liris.cnrs.fr>
 
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIEDi
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <chrono>
#include <ctime>
#include <thread>
//Command-line parsing
#include "CLI11.hpp"

//Image filtering and I/O
#define cimg_display 0
#include "CImg.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//Global flag to silent verbose messages
bool silent;

void slicedTransfer(std::vector<float> &source,
                    const std::vector<float> &target,
                    const int nbSteps,
                    const int batchSize,
                    const double factor)
{
  //Random generator init to draw random line directions
  std::mt19937 gen;
  gen.seed(10);
  std::normal_distribution<float> dist{0.0,1.0};
  
  auto N = source.size()/3;
  
  //Advection vector
  std::vector<float> advect(3*N, 0.0);
  
  //To store the 1D projections
  std::vector<float> projsource(N);
  std::vector<float> projtarget(N);
  
  //Pixel Id
  std::vector<unsigned int> idSource(N);
  std::vector<unsigned int> idTarget(N);
  
  //Lambda expression for the comparison of points in RGB
  //according to their projections
  auto lambdaProjSource = [&projsource](unsigned int a, unsigned int b) {return projsource[a] < projsource[b]; };
  auto lambdaProjTarget = [&projtarget](unsigned int a, unsigned int b) {return projtarget[a] < projtarget[b]; };
  
  for(auto i=0; i < idSource.size() ; ++i)
  {
    idSource[i]=i;
    idTarget[i]=i;
  }
  
  for(auto step =0 ; step < nbSteps; ++step)
  {
    for(auto batch = 0; batch < batchSize; ++batch )
    {
      //Random direction
      float dirx = dist(gen);
      float diry = dist(gen);
      float dirz = dist(gen);
      float norm = sqrt(dirx*dirx + diry*diry + dirz*dirz);
      dirx /= norm;
      diry /= norm;
      dirz /= norm;
      if (!silent) std::cout<<"Slice "<<step<<" batch "<<batch<<"  "<<dirx<<","<<diry<<","<<dirz<<std::endl;
      
      //We project the points
      for(auto i = 0; i < projsource.size(); ++i)
      {
        projsource[i] = dirx * source[3*i] + diry * source[3*i+1] + dirz * source[3*i+2];
        projtarget[i] = dirx * target[3*i] + diry * target[3*i+1] + dirz * target[3*i+2];
      }
      
      //1D optimal transport of the projections with two sorts
      std::thread threadA([&]{ std::sort(idSource.begin(), idSource.end(), lambdaProjSource); });
      std::sort(idTarget.begin(), idTarget.end(), lambdaProjTarget);
      threadA.join();
      
      //We accumulate the displacements in a batch
      for(auto i = 0; i < idSource.size(); ++i)
      {
        auto pix = idSource[i];
        advect[3*pix]   += dirx * (projtarget[idTarget[i]] - projsource[idSource[i]]);
        advect[3*pix+1] += diry * (projtarget[idTarget[i]] - projsource[idSource[i]]);
        advect[3*pix+2] += dirz * (projtarget[idTarget[i]] - projsource[idSource[i]]);
      }
    }
    
    //Advection
    for(auto i = 0; i <3*N; ++i)
    {
      source[i] += factor*advect[i]/(float)batchSize;
      advect[i] = 0.0;
    }
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
  double factor = 1.0;
  app.add_option("--factor", factor, "Displacement factor [0:1]");
  CLI11_PARSE(app, argc, argv);
  
  //Image loading
  int width,height, nbChannels;
  unsigned char *source = stbi_load(sourceImage.c_str(), &width, &height, &nbChannels, 0);
  if (!silent) std::cout<< "Source image: "<<width<<"x"<<height<<"   ("<<nbChannels<<")"<< std::endl;
  int width_target,height_target, nbChannels_target;
  unsigned char *target = stbi_load(targetImage.c_str(), &width_target, &height_target, &nbChannels_target, 0);
  if (!silent) std::cout<< "Target image: "<<width_target<<"x"<<height_target<<"   ("<<nbChannels_target<<")"<< std::endl;
  
  if ((width*height) != (width_target*height_target))
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
  for(auto i = 0 ; i <width*height*nbChannels; ++i)
  {
    sourcefloat[i] = static_cast<float>(source[i]);
    targetfloat[i] = static_cast<float>(target[i]);
  }
  
  //Main computation
  auto start = std::chrono::system_clock::now();
  
  slicedTransfer(sourcefloat, targetfloat, nbSteps, batchSize, factor);
  
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
  << "elapsed time: " << elapsed_seconds.count() << "s\n";

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
