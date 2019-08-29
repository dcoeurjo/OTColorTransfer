#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <chrono>
#include <ctime>

//Command-line parsing
#include "CLI11.hpp"

//Image filtering and I/O
#define cimg_display 0
#include "CImg.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "UnbalancedSliced/UnbalancedSliced.h"

//Global flag to silent verbose messages
bool silent;

void slicedTransfer(std::vector<float> &source,
                    const std::vector<float> &target,
                    const int nbSteps)
{
  omp_set_nested(0);
  
  auto N = source.size()/3;
  auto N2= target.size()/3;
  //Creating the diracs
  std::vector<std::vector<Point<3, float> > > points(2);
  points[0].resize(N);
  points[1].resize(N2);
  for (int i = 0; i < N; i++) {
    points[0][i][0] = source[i * 3] ;
    points[0][i][1] = source[i * 3+1] ;
    points[0][i][2] = source[i * 3+2] ;
  }
  for (int i = 0; i < N2; i++) {
    points[1][i][0] = target[i * 3] ;
    points[1][i][1] = target[i * 3 + 1] ;
    points[1][i][2] = target[i * 3 + 2] ;
  }
  
  //Main computation
  UnbalancedSliced sliced;

  auto start = std::chrono::system_clock::now();
  
  sliced.correspondencesNd<3, float>(points[0], points[1], nbSteps, true);
  
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
  
  //Copyback
  for (int i = 0; i < N; i++)
  {
    source[i * 3]   = points[0][i][0]  ;
    source[i * 3+1] = points[0][i][1] ;
    source[i * 3+2] = points[0][i][2];
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
  
  if ((width*height) > (width_target*height_target))
  {
    std::cout<< "The source image must be smaller (or equal to) than the target image. "<<std::endl;
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
  slicedTransfer(sourcefloat, targetfloat, nbSteps);
  
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
