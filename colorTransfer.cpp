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

//Global flag to silent verbose messages
bool silent;

void slicedTransfer(std::vector<double> &source,
                    const std::vector<double> &target,
                    const int width,
                    const int height,
                    const int nbSteps,
                    const int batchSize)
{
  //Random generator init to draw random line directions
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> dist{0.0,1.0};
  
  //Advection vector
  std::vector<double> advect(3*width*height, 0.0);
  
  //To store the 1D projections
  std::vector<double> projsource(width*height);
  std::vector<double> projtarget(width*height);
  
  //Pixel Id
  std::vector<unsigned int> idSource(width*height);
  std::vector<unsigned int> idTarget(width*height);
  
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
      double dirx = dist(gen);
      double diry = dist(gen);
      double dirz = dist(gen);
      double norm = sqrt(dirx*dirx + diry*diry + dirz*dirz);
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
      std::sort(idSource.begin(), idSource.end(), lambdaProjSource);
      std::sort(idTarget.begin(), idTarget.end(), lambdaProjTarget);
      
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
    for(auto i = 0; i <3*width*height; ++i)
    {
      source[i] += advect[i]/(double)batchSize;
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
  double sigmaXY = 16.0;
  app.add_option("--sigmaXY", sigmaXY, "Sigma parameter in the spatial domain for the bilateral regularization (16.0)");
  double sigmaV = 5.0;
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
  
  std::vector<double> sourcedouble(width*height*nbChannels);
  std::vector<double> targetdouble(width*height*nbChannels);
#pragma omp parallel for
  for(auto i = 0 ; i <width*height*nbChannels; ++i)
  {
    sourcedouble[i] = static_cast<double>(source[i]);
    targetdouble[i] = static_cast<double>(target[i]);
  }
  
  //Main computation
  slicedTransfer(sourcedouble, targetdouble, width, height, nbSteps, batchSize);
  
  
  //Output
  std::vector<unsigned char> output(width*height*nbChannels);
  if (applyRegularization)
  {
    //Regularization of the transport plan (optional)
    // (bilateral filter of the difference)
    if (!silent) std::cout<<"Applying regularization step"<<std::endl;
    cimg_library::CImg<double> transport(width, height, 1, 3);
    for(auto i=0; i<width*height; ++i)
    {
      transport[i] = sourcedouble[3*i] - static_cast<double>(source[3*i]);
      transport[i+ width*height] = sourcedouble[3*i+1] - static_cast<double>(source[3*i+1]);
      transport[i+2*width*height] = sourcedouble[3*i+2] - static_cast<double>(source[3*i+2]);
    }
    transport.blur_bilateral(transport, sigmaXY,sigmaV);
  
    auto output2(output);
    for(auto i = 0 ; i < width*height ; ++i)
    {
      output[3*i]   = static_cast<unsigned char>(  std::min(255.0, std::max(0.0, static_cast<double>(source[3*i  ]) + transport[i])));
      output[3*i+1] = static_cast<unsigned char>(  std::min(255.0, std::max(0.0, static_cast<double>(source[3*i+1]) + transport[i+ width*height])));
      output[3*i+2] = static_cast<unsigned char>(  std::min(255.0, std::max(0.0, static_cast<double>(source[3*i+2]) + transport[i+ width*height*2])));
    }
  }
  else
  {
    for(auto i = 0 ; i < width*height*nbChannels ; ++i)
      output[i] = static_cast<unsigned char>(  std::min(255.0, std::max(0.0,  sourcedouble[i])));
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
