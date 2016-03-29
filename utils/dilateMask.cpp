///
/// Cuts image by tiles (by the corresponded mask) and saves it in folders positive/ and negative/. Adds file list text file.
/// If mask provided then it used as negative. Else use bounding box as negative.
///
#include <iostream>
#include <fstream>
#include <omp.h>

#include <itkImage.h>
#include <itkMetaImageIOFactory.h>
#include <itkNrrdImageIOFactory.h>
#include <itkMinimumMaximumImageCalculator.h>
#include "itkBinaryDilateImageFilter.h"

#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkCommandLineArgumentParser.h"

int main(int argc, char* argv[])
{
  using namespace agtk;

  itk::MetaImageIOFactory::RegisterOneFactory();
  itk::NrrdImageIOFactory::RegisterOneFactory();
  std::cout << "nrrd factory registered" << std::endl;

  typedef std::list<itk::LightObject::Pointer> RegisteredObjectsContainerType;
  RegisteredObjectsContainerType registeredIOs = itk::ObjectFactoryBase::CreateAllInstance("itkImageIOBase");
  std::cout << "there are " << registeredIOs.size() << " IO objects available to the ImageFileReader." << std::endl;

  /** Create a command line argument parser. */
  CommandLineArgumentParser::Pointer parser = CommandLineArgumentParser::New();
  parser->SetCommandLineArguments(argc, argv);

  std::string maskName;
  parser->GetValue("-maskName", maskName);

  std::string listFile;
  parser->GetValue("-listFile", listFile); // conains pathes without slashes on the end

  int radius; // for padding
  parser->GetValue("-radius", radius);

  std::cout << "list file  " << listFile << std::endl;
  std::cout << "maskName  " << maskName << std::endl;
  std::cout << "radius " << radius << std::endl;

  std::vector<std::string> inputDirs;

  ifstream infile(listFile);
  std::string line;

  while (std::getline(infile, line)) {
    inputDirs.push_back(line);
  }
  std::cout << "inputData.size() " << inputDirs.size() << std::endl;

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
  omp_set_num_threads(24); // Use half threads

#pragma omp parallel for
  for (int iImage = 0; iImage < inputDirs.size(); ++iImage) {
    auto inputDir = inputDirs[iImage];
    auto maskFile = inputDir + "/" + maskName;
    auto outputFile = inputDir + "/" + maskName + "-dilate-r" + std::to_string(radius) + ".nrrd";

    std::cout << "maskFile " << maskFile << std::endl;

    // read images
    BinaryImage3D::Pointer mask = BinaryImage3D::New();
    std::cout << "load mask" << std::endl;
    if (!readImage(mask, maskFile)) {
      std::cout << "can't read " << maskFile << std::endl;
      continue;
    }

    typedef itk::MinimumMaximumImageCalculator <BinaryImage3D> ImageCalculatorFilterType;
    ImageCalculatorFilterType::Pointer imageCalculatorFilter = ImageCalculatorFilterType::New();
    imageCalculatorFilter->SetImage(mask);
    imageCalculatorFilter->ComputeMaximum();

    typedef itk::BinaryBallStructuringElement<BinaryImage3D::PixelType, BinaryImage3D::ImageDimension> StructuringElementType;
    StructuringElementType structuringElement;
    StructuringElementType::SizeType radius3D = { radius, radius, radius * mask->GetSpacing()[0] / mask->GetSpacing()[2] };
    std::cout << "radius 3d: " << radius3D << std::endl;

    structuringElement.SetRadius(radius3D);
    structuringElement.CreateStructuringElement();

    typedef itk::BinaryDilateImageFilter <BinaryImage3D, BinaryImage3D, StructuringElementType> BinaryDilateImageFilterType;

    BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
    dilateFilter->SetInput(mask);
    dilateFilter->SetDilateValue(imageCalculatorFilter->GetMaximum());
    dilateFilter->SetKernel(structuringElement);
    dilateFilter->Update();

    writeImage(dilateFilter->GetOutput(), outputFile);
  }
  return EXIT_SUCCESS;
};
