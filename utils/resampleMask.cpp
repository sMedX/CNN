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

#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkCommandLineArgumentParser.h"
#include <agtkResampling.h>

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

  float spacing; // for padding
  parser->GetValue("-radius", spacing);

  std::cout << "list file  " << listFile << std::endl;
  std::cout << "maskName  " << maskName << std::endl;
  std::cout << "spacing " << spacing << std::endl;

  std::vector<std::string> inputDirs;

  std::ifstream infile(listFile);
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
    auto outputFile = inputDir + "/" + maskName + "-s" + std::to_string(spacing) + ".nrrd";

    std::cout << "maskFile " << maskFile << std::endl;

    // read images
    BinaryImage3D::Pointer mask = BinaryImage3D::New();
    std::cout << "load mask" << std::endl;
    if (!readImage(mask, maskFile)) {
      std::cout << "can't read " << maskFile << std::endl;
      continue;
    }

    agtk::Image3DSpacing spacing3D;
    spacing3D[0] = spacing;
    spacing3D[1] = spacing;
    spacing3D[2] = mask->GetSpacing()[2];

    mask = resamplingBinary(mask.GetPointer(), spacing3D);
    writeImage(mask, outputFile);
  }
  return EXIT_SUCCESS;
};
