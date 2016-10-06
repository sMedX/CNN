#include <iostream>
#include <fstream>
#include <omp.h>

#include <boost/filesystem.hpp>

#include <itkImage.h>
#include <itkMetaImageIOFactory.h>
#include <itkNrrdImageIOFactory.h>
#include <itkTestingExtractSliceImageFilter.h>

#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkCommandLineArgumentParser.h"

#include "preprocess.h"
#include <agtkBinaryImageUtilities.h>

namespace fs = boost::filesystem;

template <typename TPixel>
itk::SmartPointer<itk::Image<TPixel, 2>> exctractSlice(itk::Image<TPixel, 3> * image16, int z)
{
  typedef itk::Image<TPixel, 3> ImageType;
  typename ImageType::RegionType desiredRegion({ 0, 0, z }, { 512, 512, 0 });

  typedef itk::Testing::ExtractSliceImageFilter<ImageType, itk::Image<TPixel, 2> > FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetExtractionRegion(desiredRegion);
  filter->SetInput(image16);
#if ITK_VERSION_MAJOR >= 4
  filter->SetDirectionCollapseToIdentity(); // This is required.
#endif
  filter->Update();

  return filter->GetOutput();
}

int main(int argc, char* argv[])
{
  using namespace agtk;
  using namespace caffefication;

  const std::string ext = ".png";

  const std::string positive = "pos", negative = "neg"; //for 2classes

  itk::MetaImageIOFactory::RegisterOneFactory();
  itk::NrrdImageIOFactory::RegisterOneFactory();
  std::cout << "nrrd factory registered" << std::endl;

  typedef std::list<itk::LightObject::Pointer> RegisteredObjectsContainerType;
  RegisteredObjectsContainerType registeredIOs = itk::ObjectFactoryBase::CreateAllInstance("itkImageIOBase");
  std::cout << "there are " << registeredIOs.size() << " IO objects available to the ImageFileReader." << std::endl;

  /** Create a command line argument parser. */
  CommandLineArgumentParser::Pointer parser = CommandLineArgumentParser::New();
  parser->SetCommandLineArguments(argc, argv);

  std::string imageName;
  parser->GetValue("-imageName", imageName); // patient.nrrd for example

  std::string labelName1;
  parser->GetValue("-labelName1", labelName1); //class under label '1', livertumors for example

  std::string inputFolder;
  parser->GetValue("-inFolder", inputFolder); // directory, that contains subdirectory listed in listFile

  std::string listFile;
  parser->GetValue("-listFile", listFile); // conains pathes without slashes on the end

  std::string preset;
  parser->GetValue("-preset", preset);

  std::string outputFolder;
  parser->GetValue("-outFolder", outputFolder);

  int outSize = 256;
  parser->GetValue("-outSize", outSize);

  std::cout << "list file  " << listFile << std::endl;
  std::cout << "imageName  " << imageName << std::endl;
  std::cout << "input folder  " << inputFolder << std::endl;
  std::cout << "labelName1  " << labelName1 << std::endl;
  std::cout << "output folder  " << outputFolder << std::endl;
  std::cout << "outSize  " << outSize << std::endl;
  std::cout << "preset " << preset << std::endl;

  std::vector<std::string> inputDirs;
  std::ifstream infile(listFile);
  std::string line;
  while (std::getline(infile, line)) {
    inputDirs.push_back(inputFolder + line);
  }
  std::cout << "inputData.size() " << inputDirs.size() << std::endl;

  // set up output directories
  fs::create_directories(outputFolder);

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
  omp_set_num_threads(24); // Use 24 threads

#pragma omp parallel for
  for (int iImage = 0; iImage < inputDirs.size(); ++iImage) {
    auto inputDir = inputDirs[iImage];

    // read images
    std::cout << "load image" << std::endl;
    auto imageFile = inputDir + "/" + imageName;
    Int16Image3D::Pointer image16 = Int16Image3D::New();
    if (!readImage<Int16Image3D>(image16, imageFile)) {
      std::cout << "can't read " << imageFile;
      continue;
    }

    std::cout << "load label1" << std::endl;
    auto labelFile1 = inputDir + "/" + labelName1;
    std::cout << "labelFile1 " << labelFile1 << std::endl;
    BinaryImage3D::Pointer label1 = BinaryImage3D::New();
    if (!readImage(label1, labelFile1)) {
      std::cout << "can't read " << labelFile1;
      continue;
    }

    std::cout << "preprocess image" << std::endl;
    auto image8 = smartCastImage(preset, image16, nullptr);
    std::cout << "done" << std::endl;

    itk::ImageBase<2>::SizeType outSize2D = { outSize, outSize };

    bool isValidRegion;
    auto region = getBinaryMaskBoundingBoxRegion(label1, &isValidRegion);
    if (!isValidRegion) {
      continue;
    }

    //expand region a little bit
    const int slicePadding = 2; //pad region by z axis
    const int maxSlice = image8->GetLargestPossibleRegion().GetUpperIndex()[2];
    int startSlice = region.GetIndex()[2] - slicePadding;
    if (startSlice < 0) {
      startSlice = 0;
    }
    int endSlice = region.GetUpperIndex()[2] + slicePadding;
    if (endSlice > maxSlice) {
      endSlice = maxSlice;
    }

    for (size_t z = startSlice; z <= endSlice; z++) {
      auto sliceImage = exctractSlice(image8.GetPointer(), z);
      auto sliceLabel = exctractSlice(label1.GetPointer(), z);

      sliceImage = resize(sliceImage.GetPointer(), outSize2D);
      sliceLabel = resizeBinary(sliceLabel.GetPointer(), outSize2D);

      itk::ImageRegionIterator<UInt8Image2D> it(sliceLabel, sliceLabel->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(it.Get() > 0 ? 1 : 0);
      }

      auto indexStr = "n" + std::to_string(iImage) + "_z" + std::to_string(z);
      auto dirImages = outputFolder + "/images/";
      auto dirLabels = outputFolder + "/labels/";

      fs::create_directories(dirImages);
      fs::create_directories(dirLabels);

      std::string filenameImage = dirImages + indexStr + ext;
      std::string filenameLabel = dirLabels + indexStr + ext;

      writeImage(sliceImage.GetPointer(), filenameImage);
      writeImage(sliceLabel.GetPointer(), filenameLabel);

      std::cout << z << " slice of " << iImage << " image" << std::endl;
    }
  }
  return EXIT_SUCCESS;
};
