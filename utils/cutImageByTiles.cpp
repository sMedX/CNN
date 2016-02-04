///
/// Cuts image by tiles (by the corresponded mask) and saves it in folders positive/ and negative/. Adds file list text file.
/// If mask provided then it used as negative. Else use bounding box as negative.
///
#include <iostream>
#include <fstream>
#include <omp.h>

#include <itkImage.h>
#include <itkTestingExtractSliceImageFilter.h>
#include <itkAddImageFilter.h>

#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkCommandLineArgumentParser.h"
#include "agtkBinaryImageUtilities.h"
#include "agtkResampling.h"

template <typename TPixel>
agtk::UInt8Image2D::Pointer getTile(const itk::Image<TPixel, 3>* image, const typename itk::Image<TPixel, 3>::IndexType& index, int halfSize);

int main(int argc, char* argv[])
{
  using namespace agtk;

  const std::string ext = ".png";
  const int negativeStride = 4;
  const std::string BOUNDING_BOX = "BOUNDING_BOX";
  const std::string NO_MASK = "NO_MASK";

  /** Create a command line argument parser. */
  CommandLineArgumentParser::Pointer parser = CommandLineArgumentParser::New();
  parser->SetCommandLineArguments(argc, argv);

  std::string imageName;
  parser->GetValue("-imageName", imageName);

  std::string labelName;
  parser->GetValue("-labelName", labelName);

  std::string maskName;
  parser->GetValue("-maskName", maskName);

  std::string listFile;
  parser->GetValue("-listFile", listFile); // conains pathes without slashes on the end

  int radius;
  parser->GetValue("-radius", radius);

  Image3DSize stride = { 1, 1, 0 };
  parser->GetITKValue<Image3DSize>("-stride", stride);

  std::string preset;
  parser->GetValue("-preset", preset);

  agtk::Image2DSpacing spacingXY;
  spacingXY.Fill(0);
  parser->GetITKValue("-spacingXY", spacingXY);

  std::string outputFolder;
  parser->GetValue("-folder", outputFolder);

  std::cout << "list file  " << listFile << std::endl;
  std::cout << "imageName  " << imageName << std::endl;
  std::cout << "labelName  " << labelName << std::endl;
  std::cout << "maskName  " << maskName << std::endl;
  std::cout << "output folder  " << outputFolder << std::endl;

  std::cout << std::endl;
  std::cout << "radius " << radius << std::endl;
  std::cout << "stride " << stride << std::endl;
  std::cout << "preset " << preset << std::endl;
  std::cout << "spacingXY " << spacingXY << std::endl;

  std::cout << std::endl;
  std::cout << "stride for negative points is " << negativeStride << std::endl;

  bool isNoMask = false;
  bool isBoundingBox = false;
  if (maskName == NO_MASK) {
    isNoMask = true;
  } else if (maskName == BOUNDING_BOX) {
    isBoundingBox = true;
  }
  std::vector<std::string> inputDirs;

  ifstream infile(listFile);
  std::string line;

  while (std::getline(infile, line)) {
    inputDirs.push_back(line);
  }
  std::cout << "inputData.size() " << inputDirs.size() << std::endl;

  // set up output directories
  system((std::string("md ") + outputFolder).c_str());

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
  omp_set_num_threads(24); // Use 24 threads

  //int c = 0;

  #pragma omp parallel for
  for (int iImage = 0; iImage < inputDirs.size(); ++iImage) {
    auto inputDir = inputDirs[iImage];
    auto imageFile = inputDir + "/" + imageName;
    auto labelFile = inputDir + "/" + labelName;
    auto maskFile = inputDir + "/" + maskName;

    std::cout << "imageFile " << imageFile << std::endl;
    std::cout << "labelFile " << labelFile << std::endl;
    std::cout << "maskFile " << maskFile << std::endl;

    // read images
    std::cout << "load image" << std::endl;
    Int16Image3D::Pointer image16 = Int16Image3D::New();
    if (!readImage<Int16Image3D>(image16, imageFile)) {
      std::cout << "can't read " << imageFile;
      continue;
    }

    std::cout << "load label" << std::endl;
    BinaryImage3D::Pointer label = BinaryImage3D::New();
    if (!readImage(label, labelFile)) {
      std::cout << "can't read " << labelFile;
      continue;
    }

    BinaryImage3D::Pointer mask = BinaryImage3D::New();
    if (!(isBoundingBox || isNoMask)) {
      std::cout << "load mask" << std::endl;
      if (!readImage(mask, maskFile)) {
        std::cout << "can't read " << maskFile << std::endl;
        continue;
      }
    }

    std::cout << "preprocess images" << std::endl;
    std::cout << "shift, sqeeze" << std::endl;
    UInt8Image3D::Pointer image;

    if (preset == "pancreas") {
      const int shift = 190;
      const int squeeze = 2;

      // x' = (x + shift)/squeeze
      itk::ImageRegionIterator<Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set((it.Get() + shift) / squeeze);
      }
    }
    else if (preset == "livertumors") {
      const int shift = 40;

      // x' = x + shift
      itk::ImageRegionIterator<Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(it.Get() + shift);
      }
    }
    std::cout << "cast (truncate)" << std::endl;
    typedef itk::CastImageFilter<Int16Image3D, UInt8Image3D> Cast;
    auto cast = Cast::New();
    cast->SetInput(image16);
    cast->Update();
    image = cast->GetOutput();

    if (spacingXY[0] != 0) { //resample image by axial slices
      std::cout << "resample" << std::endl;
      Image3DSpacing spacing;
      spacing[0] = spacingXY[0];
      spacing[1] = spacingXY[1];
      spacing[2] = image->GetSpacing()[2];

      image = resampling(image.GetPointer(), spacing);
      label = resamplingBinary(label.GetPointer(), spacing);

      if (mask.IsNotNull()) {
        mask = resamplingBinary(mask.GetPointer(), spacing);
      }
    }

    std::cout << "calculate indices" << std::endl;
    std::vector<BinaryImage3D::IndexType> indices;

    auto shrinkRegion = image->GetLargestPossibleRegion();
    shrinkRegion.ShrinkByRadius(radius);

    int negativeCount = 0;

    if (isBoundingBox) {
      auto region = getBinaryMaskBoundingBoxRegion(label);
      region.Crop(shrinkRegion);
      std::cout << "use bounding box: " << region << std::endl;

      itk::ImageRegionConstIterator<BinaryImage3D> it(label, region);

      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        if (label->GetPixel(it.GetIndex()) == 0) {// if negative
          if (negativeCount % negativeStride == 0) {
            indices.push_back(it.GetIndex());
          }
          negativeCount++;
        }
        else {
          indices.push_back(it.GetIndex());
        }
      }
    }
    else if (isNoMask) { // this part can be removed when ordinal white image will be used as mask
      std::cout << "don't mask" << std::endl;
      itk::ImageRegionConstIterator<BinaryImage3D> it(label, shrinkRegion);
      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        if (label->GetPixel(it.GetIndex()) == 0) {// if negative
          if (negativeCount % negativeStride == 0) {
            indices.push_back(it.GetIndex());
          }
          negativeCount++;
        }
        else {
          indices.push_back(it.GetIndex());
        }
      }
    } else {
      std::cout << "use mask" << std::endl;
      itk::ImageRegionConstIterator<BinaryImage3D> itMask(mask, shrinkRegion);

      for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
        if (itMask.Get() != 0) {
          if (label->GetPixel(itMask.GetIndex()) == 0) {// if negative
            if (negativeCount % negativeStride == 0) {
              indices.push_back(itMask.GetIndex());
            }
            negativeCount++;
          }
          else {
            indices.push_back(itMask.GetIndex());
          }
        }
      }
    }
    const int totalCount = indices.size();
    negativeCount /= negativeStride;

    std::cout << "positiveCount: " << totalCount - negativeCount << std::endl;
    std::cout << "negativeCount: " << negativeCount << std::endl;
    std::cout << "totalCount: " << totalCount << std::endl;

    // set up output directories
    std::string iImageStr = inputDir.substr(inputDir.length() - 3);//take 3 last chars, e.g. 012
    const std::string pos = "tum", neg = "notum";
    std::string outDir = outputFolder + "\\" + iImageStr + "\\";
    system((std::string("md ") + outDir).c_str());

    system((std::string("md ") + outDir + pos).c_str());
    system((std::string("md ") + outDir + neg).c_str());

    for (int j = 0; j < totalCount; ++j) {
      auto& index = indices[j];
      auto tile = getTile<UInt8Pixel>(image, index, radius);

      //save image
      std::string indexStr = std::to_string(index[0]) + "_" + std::to_string(index[1]) + "_" + std::to_string(index[2]);
      auto labelI = label->GetPixel(index);
      std::string labelStr = labelI == 0 ? "notum" : "tum";
      std::string filename = outDir + labelStr + "\\" + indexStr + ext;

      writeImage(tile.GetPointer(), filename);

      if (j % 10000 == 0) {
        std::cout << static_cast<double>(j*100)/totalCount << "% of " << iImageStr << " image" << std::endl;
      }
    }

  }


  return EXIT_SUCCESS;
};

template <typename TPixel>
agtk::UInt8Image2D::Pointer getTile(const itk::Image<TPixel, 3>* image, const typename itk::Image<TPixel, 3>::IndexType& index, int halfSize)
{
  typedef itk::Image<TPixel, 3> ImageType3D;

  typedef itk::Testing::ExtractSliceImageFilter<ImageType3D, agtk::UInt8Image2D> ExtractVolumeFilterType;

  auto extractVolumeFilter = ExtractVolumeFilterType::New();
  agtk::Image3DSize size = { 2 * halfSize, 2 * halfSize, 0 };
  agtk::Image3DIndex start = { index[0] - halfSize + 1, index[1] - halfSize + 1, index[2] };

  typename ImageType3D::RegionType outputRegion;
  outputRegion.SetSize(size);
  outputRegion.SetIndex(start);

  extractVolumeFilter->SetInput(image);
  extractVolumeFilter->SetExtractionRegion(outputRegion);
  extractVolumeFilter->SetDirectionCollapseToGuess();
  extractVolumeFilter->Update();

  return extractVolumeFilter->GetOutput();
}
