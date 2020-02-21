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
#include <sstream>
#include <assert.h>
//Command-line parsing
#include "CLI11.hpp"

//Image filtering and I/O
#define cimg_display 0  
#include "CImg.h"

//Global flag to silent verbose messages
bool silent;

typedef std::vector<double> Point;
typedef std::vector<Point> PointSet;

PointSet loadPointset(const std::string &filename)
{
  PointSet output;
  std::string line;
  std::ifstream ifs(filename,std::ifstream::in);
  auto dim=0;
  while( std::getline( ifs, line ) ) // read the file one line at a time
  {
    Point p;
    std::istringstream iss(line);
    std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                     std::istream_iterator<std::string>());
    assert( (dim==0) || (dim = results.size()));
    dim = results.size();
    
    for(auto s: results)
        p.push_back(std::stod(s));
    output.push_back(p);
  }
  ifs.close();
  std::cout<<"NbPoints = "<< output.size()<< " dimension = "<<output[0].size()<<std::endl;
  return output;
}



void regularize(const PointSet &sourceOrig,
                PointSet &source,
                const std::vector<unsigned int> dims,
                const double sigmaXY,
                const double sigmaV)
{
  
  unsigned int size = (int)sqrt(source.size());
  
  for(auto i=0 ; i < dims.size(); ++i)
  {
    cimg_library::CImg<double> transport(size, size, 1, 1);
    for(auto j=0; j<size*size; ++j)
      transport[j] = source[j][dims[i]] - sourceOrig[j][dims[i]];;
    transport.blur_bilateral(transport, sigmaXY,sigmaV);
    
    for(auto j = 0 ; j < size*size ; ++j)
      source[j][dims[i]] = std::min(1.0, std::max(0.0, sourceOrig[j][dims[i]] + transport[j]));
    
  }
}


void dumpPointset(const std::string &filename, PointSet &pset)
{
  std::ofstream ofs(filename,std::ofstream::out);
  
  for(auto &p: pset)
  {
    for(auto &v: p)
      ofs <<v<<" ";
    ofs<<std::endl;
  }
  ofs.close();
}


/// Dot product for specified dimensions
/// @param a a point
/// @param b a second point
/// @param dims the dimensions (subspace for the dot product)
/// @return the dot product
double dot(const Point &a, const Point &dir, const std::vector<unsigned int> &dims)
{
  double res=0.0;
  for (auto i = 0; i < dims.size(); ++i)
    res+= a[ dims[i] ] * dir[ dims[i] ];
  return res;
}



void slicedTransfer(PointSet &source,
                    const PointSet &target,
                    const std::vector<unsigned int> &dims,
                    const int nbSteps,
                    const int batchSize)
{
  
  //Random generator init to draw random line directions
  std::mt19937 gen;
  gen.seed(10);
  std::normal_distribution<double> dist{0.0,1.0};
  std::uniform_real_distribution<double> unif(0.0,1.0);
  auto N = source.size();
  
  assert(source.size()==target.size());
  
  //Advection vector
  Point zero(source[0].size(), 0.0);
  PointSet advect(N, zero);
  
  //To store the 1D projections
  std::vector<double> projsource(N);
  std::vector<double> projtarget(N);
  //point Id
  std::vector<unsigned int> idSource(N);
  std::vector<unsigned int> idTarget(N);
  
  //Lambda expression for the comparison of points in RGB
  //according to their projections
  auto lambdaProjSource = [&projsource](unsigned int a, unsigned int b) {return projsource[a] < projsource[b]; };
  auto lambdaProjTarget = [&projtarget](unsigned int a, unsigned int b) {return projtarget[a] < projtarget[b]; };
  
  for(auto i=0; i <N ; ++i)
  {
    idSource[i]=i;
    idTarget[i]=i;
  }
  
  for(auto step =0 ; step < nbSteps; ++step)
  {
    for(auto batch = 0; batch < batchSize; ++batch )
    {
      //Random direction
      Point directions( source[0].size() , 0.0 );
      double norm=0.0;
     
      if (dims.size()==1)
        directions[dims[0]] = unif(gen);
      else
      {
        for(auto i = 0; i < dims.size(); ++i  )
        {
          directions[dims[i]] = dist(gen);
          norm += directions[dims[i]] * directions[dims[i]];
        }
        norm = std::sqrt(norm);
        for(auto i = 0; i < dims.size(); ++i  )
          directions[dims[i]] /= norm;
      }
     
     if (!silent)
      {
        std::cout<<"Slice "<<step<<" batch "<<batch<<"  --  ";
        for(auto i = 0; i < directions.size(); ++i  )
          std::cout<<directions[i]<<" ";
        std::cout << std::endl;
      }
      
      //We project the points
      //1D optimal transport of the projections with two sorts
      std::thread threadA([&]{for(auto i = 0; i < N; ++i)
                                 projsource[i] = dot(source[i], directions, dims);
                              std::sort(idSource.begin(), idSource.end(), lambdaProjSource); });
      
      //Parallel
      for(auto i = 0; i <N; ++i)
        projtarget[i] = dot(target[i], directions, dims);
      std::sort(idTarget.begin(), idTarget.end(), lambdaProjTarget);
      threadA.join();
      
      //We accumulate the displacements in a batch
      for(auto p = 0; p < N; ++p)
      {
        auto pix = idSource[p];
        for(auto i = 0; i < dims.size(); ++i  )
          advect[pix][dims[i]] += directions[dims[i]]  * (projtarget[idTarget[p]] - projsource[ pix ]);
      }
    }
    for(auto i = 0; i <N; ++i)
    {
      for(auto k=0; k < dims.size(); ++k)
      {
        source[i][dims[k]] += advect[i][dims[k]]/(double)batchSize;
        advect[i][dims[k]] = 0.0;
      }
    }
  }
}




int main(int argc, char **argv)
{
  CLI::App app{"colorTransfer"};
  std::string sourceImage;
  app.add_option("-s,--source", sourceImage, "Source file") ->required()->check(CLI::ExistingFile);
  std::string targetImage;
  app.add_option("-t,--target", targetImage, "Target file")->required()->check(CLI::ExistingFile);
  std::string outputImage;
  app.add_option("-o,--output", outputImage, "Output file")->required();
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
 
  
  std::vector<unsigned int> dimensions;
  app.add_option("--dims", dimensions, "OT subspace");
  CLI11_PARSE(app, argc, argv);
  
 
  //Loading data
  PointSet source = loadPointset(sourceImage);
  PointSet orig = source;
  PointSet target = loadPointset(targetImage);
  
  slicedTransfer(source, target, dimensions, nbSteps, batchSize);

  if (applyRegularization)
  {
    if (!silent) std::cout<<"Applying regularization step"<<std::endl;
    regularize(orig, source, dimensions, sigmaXY, sigmaV);
  }
  //export
  dumpPointset(outputImage, source);

  exit(0);
}
