#include <iostream>
#include <string>

#include "agtkCommandLineArgumentParser.h"
#include "agtkTypes.h"
#include "agtkIO.h"
#include <itkDiscreteGaussianImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>

using namespace agtk;

int main(int argc, char** argv)
{
  /** Create a command line argument parser. */
  CommandLineArgumentParser::Pointer parser = CommandLineArgumentParser::New();

  parser->SetCommandLineArguments(argc, argv);

  std::string imageFile;
  parser->GetValue("-image", imageFile);

  float gaussianVariance;
  parser->GetValue("-gaussianVariance", gaussianVariance);

  std::cout << "image file  " << imageFile << std::endl;

  // read images
  FloatImage3D::Pointer image = FloatImage3D::New(); //float is special
  if (!readImage<FloatImage3D>(image, imageFile)) {
    return EXIT_FAILURE;
  }

  // blur
  // Create and setup a Gaussian filter
  typedef itk::DiscreteGaussianImageFilter<FloatImage3D, FloatImage3D >  filterType;
  filterType::Pointer gaussianFilter = filterType::New();
  gaussianFilter->SetInput(image);
  gaussianFilter->SetVariance(gaussianVariance);

  //treshold
  typedef itk::BinaryThresholdImageFilter<FloatImage3D, BinaryImage3D> BinaryThresholdImageFilterType;
  BinaryThresholdImageFilterType::Pointer thresholdFilter = BinaryThresholdImageFilterType::New();
  thresholdFilter->SetInput(gaussianFilter->GetOutput());
  thresholdFilter->SetLowerThreshold(0.5);
  //thresholdFilter->SetUpperThreshold(upperThreshold);
  thresholdFilter->SetInsideValue(1);
  thresholdFilter->SetOutsideValue(0);

  //save
  writeImage(thresholdFilter->GetOutput(), imageFile + "gaussian" + std::to_string(gaussianVariance) + ".nrrd");
  return EXIT_SUCCESS;
}