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
#include "agtkDistanceBasedMetrics.h"

using namespace agtk;

void writeReport(std::string& imageFile, std::string& reportFile,
  DistanceBasedMetrics<itk::Image<unsigned char, 3>, itk::Image<unsigned char, 3>>::Pointer distBasedMetrics,
  ConfusionMatrixBasedMetrics<itk::Image<unsigned char, 3>, itk::Image<unsigned char, 3>>::Pointer confBasedMetrics)
{
  std::string dlm = ";";

  std::string header = dlm;

  int idx1 = imageFile.find_last_of("\\/");
  int idx2 = imageFile.find_last_of(".");
  std::string scores = imageFile.substr(idx1 + 1, idx2 - idx1 - 1) + dlm;

  header += "ASD, mm" + dlm;
  scores += std::to_string(distBasedMetrics->GetAverageDistance()) + dlm;
  header += "RMSD, mm" + dlm;
  scores += std::to_string(distBasedMetrics->GetRootMeanSquareDistance()) + dlm;
  header += "Hausdorff, mm" + dlm;
  scores += std::to_string(distBasedMetrics->GetHausdorffDistance()) + dlm;

  header += "True Positive" + dlm;
  scores += std::to_string(confBasedMetrics->GetTruePositive()) + dlm;
  header += "False Negative" + dlm;
  scores += std::to_string(confBasedMetrics->GetFalseNegative()) + dlm;
  header += "False Positive" + dlm;
  scores += std::to_string(confBasedMetrics->GetFalsePositive()) + dlm;
  header += "True Negative" + dlm;
  scores += std::to_string(confBasedMetrics->GetTrueNegative()) + dlm;

  header += "TruePositiveRate" + dlm;
  scores += std::to_string(confBasedMetrics->GetTruePositiveRate()) + dlm;
  header += "TrueNegativeRate" + dlm;
  scores += std::to_string(confBasedMetrics->GetTrueNegativeRate()) + dlm;
  header += "PositivePredictiveRate" + dlm;
  scores += std::to_string(confBasedMetrics->GetPositivePredictiveRate()) + dlm;
  header += "NegativePredictiveRate" + dlm;
  scores += std::to_string(confBasedMetrics->GetNegativePredictiveRate()) + dlm;
  header += "AccuracyValue" + dlm;
  scores += std::to_string(confBasedMetrics->GetAccuracyValue()) + dlm;

  header += "VOE" + dlm;
  scores += std::to_string(confBasedMetrics->GetVolumeOverlapError()) + dlm;
  header += "DICE" + dlm;
  scores += std::to_string(confBasedMetrics->GetDiceCoefficient()) + dlm;
  header += "Jaccard" + dlm;
  scores += std::to_string(confBasedMetrics->GetJaccardCoefficient()) + dlm;

  bool fileExist = boost::filesystem::exists(reportFile);

  std::ofstream ofile;
  ofile.open(reportFile, std::ofstream::out | std::ofstream::app);

  if (!fileExist) {
    ofile << header << std::endl;
  }

  ofile << scores << std::endl;
  ofile.close();
}

int main(int argc, char** argv)
{
  /** Create a command line argument parser. */
  CommandLineArgumentParser::Pointer parser = CommandLineArgumentParser::New();

  parser->SetCommandLineArguments(argc, argv);

  std::string imageFile;
  parser->GetValue("-testImage", imageFile);

  std::string labelFile;
  parser->GetValue("-label", labelFile);

  std::string reportFile;
  parser->GetValue("-report", reportFile);

  std::cout << "test image file  " << imageFile << std::endl;
  std::cout << "label file " << labelFile << std::endl;
  std::cout << "report file: " << reportFile << std::endl;

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
    testImage = resamplingLike(testImage.GetPointer(), labelImage.GetPointer());
  }
  // Evaluate distance based metrics
  std::cout << std::endl;
  std::cout << "Evaluate distance based metrics..." << std::endl;
  typedef DistanceBasedMetrics<BinaryImage3D, BinaryImage3D> DistanceMetricType;
  DistanceMetricType::Pointer distBasedMetrics = DistanceMetricType::New();
  distBasedMetrics->SetFixedImage(labelImage);
  distBasedMetrics->SetMovingImage(testImage);
  
  try {
    distBasedMetrics->Evaluate();
  } catch (const itk::ExceptionObject& ex) {
    ex.Print(std::cout);
    return EXIT_FAILURE;
  }

  distBasedMetrics->PrintReport(std::cout);

  // Evaluate confusion matrix based metrics
  std::cout << std::endl;
  std::cout << "Evaluate confusion matrix based metrics..." << std::endl;

  typedef ConfusionMatrixBasedMetrics<BinaryImage3D, BinaryImage3D> ConfusionMatrixType;
  ConfusionMatrixType::Pointer confBasedMetrics = ConfusionMatrixType::New();
  confBasedMetrics->SetFixedImage(labelImage);
  confBasedMetrics->SetMovingImage(testImage);
  
  try {
    confBasedMetrics->Evaluate();
  }
  catch (const itk::ExceptionObject& ex) {
    ex.Print(std::cout);
    return EXIT_FAILURE;
  }

  confBasedMetrics->PrintReport(std::cout);

  writeReport(imageFile, reportFile, distBasedMetrics, confBasedMetrics);

  return EXIT_SUCCESS;
}