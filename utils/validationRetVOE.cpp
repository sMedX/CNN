#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <boost/filesystem.hpp>

#include "agtkCommandLineArgumentParser.h"
#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkResampling.h"

#include "agtkConfusionMatrixBasedMetrics.h"

using namespace agtk;

int main(int argc, char** argv)
{
  /** Create a command line argument parser. */
  CommandLineArgumentParser::Pointer parser = CommandLineArgumentParser::New();

  parser->SetCommandLineArguments(argc, argv);

  std::string imageFile;
  parser->GetValue("-testImage", imageFile);

  std::string labelFile;
  parser->GetValue("-label", labelFile);

  
  std::cout << "test image file  " << imageFile << std::endl;
  std::cout << "label file " << labelFile << std::endl;

  // read images
  BinaryImage3D::Pointer testImage = BinaryImage3D::New();
  if (!readImage<BinaryImage3D>(testImage, imageFile)) {
    return EXIT_FAILURE;
  }

  BinaryImage3D::Pointer labelImage = BinaryImage3D::New();
  if (!readImage<BinaryImage3D>(labelImage, labelFile)) {
    return EXIT_FAILURE;
  }

  // todo decide who must make resampling to initial spacing validator or segmentator. prefer segmentator
  if (testImage->GetSpacing() != labelImage->GetSpacing()) {
    std::cout << "Resampling test image" << std::endl;
    testImage = resample(testImage.GetPointer(), labelImage.GetPointer());
  }
  // Evaluate distance based metrics
  typedef ConfusionMatrixBasedMetrics<BinaryImage3D, BinaryImage3D> ConfusionMatrixType;
  ConfusionMatrixType::Pointer confBasedMetrics = ConfusionMatrixType::New();
  confBasedMetrics->SetFixedImage(labelImage);
  confBasedMetrics->SetMovingImage(testImage);
  confBasedMetrics->Evaluate();

  return int(confBasedMetrics->GetVolumeOverlapError() * 100);
}