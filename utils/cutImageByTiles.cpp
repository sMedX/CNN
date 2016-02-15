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
#include <itkMetaImageIOFactory.h>
#include <itkNrrdImageIOFactory.h>

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

  const std::string TP = "TP", TN = "TN", FP = "FP", FN = "FN"; //for 4classes
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
  parser->GetValue("-imageName", imageName);

  std::string labelName1;
  parser->GetValue("-labelName1", labelName1); //class under label '1'

  std::string labelName2; //class under label '2' that is entirely in class '1', optional
  parser->GetValue("-labelName2", labelName2);

  std::string maskName;
  parser->GetValue("-maskName", maskName);

  std::string adaptiveName; //name of adaptive image i.e. result of previous classification
  parser->GetValue("-adaptive", adaptiveName);

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
  std::cout << "labelName1  " << labelName1 << std::endl;
  std::cout << "labelName2  " << labelName2 << std::endl;
  std::cout << "maskName  " << maskName << std::endl;
  std::cout << "adaptiveName  " << adaptiveName << std::endl;
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
    auto imageFile = inputDir + "\\" + imageName;
    auto labelFile1 = inputDir + "\\" + labelName1;
    auto labelFile2 = inputDir + "\\" + labelName2;
    auto maskFile = inputDir + "\\" + maskName;
    auto adaptiveFile = inputDir + "\\" + adaptiveName;

    std::cout << "imageFile " << imageFile << std::endl;
    std::cout << "labelFile1 " << labelFile1 << std::endl;
    std::cout << "labelFile2 " << labelFile2 << std::endl;
    std::cout << "maskFile " << maskFile << std::endl;
    std::cout << "adaptiveFile " << adaptiveFile << std::endl;

    bool is4Classes = !adaptiveName.empty();

    // read images
    std::cout << "load image" << std::endl;
    Int16Image3D::Pointer image16 = Int16Image3D::New();
    if (!readImage<Int16Image3D>(image16, imageFile)) {
      std::cout << "can't read " << imageFile;
      continue;
    }

    std::cout << "load label1" << std::endl;
    BinaryImage3D::Pointer label1 = BinaryImage3D::New();
    if (!readImage(label1, labelFile1)) {
      std::cout << "can't read " << labelFile1;
      continue;
    }

    std::cout << "load label2" << std::endl;
    BinaryImage3D::Pointer label2 = BinaryImage3D::New();
    if (!readImage(label2, labelFile2)) {
      std::cout << "can't read " << labelFile2 << ". label2 will not be used." << std::endl;
      label2->CopyInformation(label1); // make an empty image instead
      label2->Allocate();
      label2->FillBuffer(0);
    }

    BinaryImage3D::Pointer mask = BinaryImage3D::New();
    if (!(isBoundingBox || isNoMask)) {
      std::cout << "load mask" << std::endl;
      if (!readImage(mask, maskFile)) {
        std::cout << "can't read " << maskFile << std::endl;
        continue;
      }
    }

    BinaryImage3D::Pointer adaptive = BinaryImage3D::New();
    if (is4Classes) {
      std::cout << "load adaptive" << std::endl;
      if (!readImage(adaptive, adaptiveFile)) {
        std::cout << "can't read " << adaptiveFile << std::endl;
        continue;
      }
    }

    //todo add check for equal size

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
      UInt8Image3D::PixelType minValue = 0, maxValue = 255;

      // x' = x + shift
      itk::ImageRegionIterator<Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        auto val = it.Get() + shift;
        if (val < minValue) {
          val = minValue;
        }
        else if (val > maxValue) {
          val = maxValue;
        }
        it.Set(val);
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
      label1 = resamplingBinary(label1.GetPointer(), spacing);
      label2 = resamplingBinary(label2.GetPointer(), spacing);

      if (mask.IsNotNull()) {
        mask = resamplingBinary(mask.GetPointer(), spacing);
      }

      if (is4Classes) {
        adaptive = resamplingBinary(adaptive.GetPointer(), spacing);
      }
    }

    std::cout << "calculate indices" << std::endl;
    std::vector<BinaryImage3D::IndexType> indices;

    const Image3DOffset radius3D = {radius, radius, 0};

    //shrink by x and y only
    Image3DIndex movedIndex = image->GetLargestPossibleRegion().GetIndex() + radius3D;
    Image3DSize shrinkedSize;
    for (size_t i = 0; i < Image3DRegion::ImageDimension; i++) {
      shrinkedSize[i] = image->GetLargestPossibleRegion().GetSize()[i] - 2 * radius3D[i];
    }
    Image3DRegion shrinkRegion = {movedIndex, shrinkedSize};

    int negativeCount = 0;
    int class1Count = 0;
    int class2Count = 0;

    // not implemented yet for 3-class problem
    //if (isBoundingBox) {
    //  auto region = getBinaryMaskBoundingBoxRegion(label);
    //  region.Crop(shrinkRegion);
    //  std::cout << "use bounding box: " << region << std::endl;

    //  itk::ImageRegionConstIterator<BinaryImage3D> it(label, region);

    //  for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
    //    if (label->GetPixel(it.GetIndex()) == 0) {// if negative
    //      if (negativeCount % negativeStride == 0) {
    //        indices.push_back(it.GetIndex());
    //      }
    //      negativeCount++;
    //    }
    //    else {
    //      indices.push_back(it.GetIndex());
    //    }
    //  }
    //}
    //else if (isNoMask) { // this part can be removed when ordinal white image will be used as mask
    //  std::cout << "don't mask" << std::endl;
    //  itk::ImageRegionConstIterator<BinaryImage3D> it(label, shrinkRegion);
    //  for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
    //    if (label->GetPixel(it.GetIndex()) == 0) {// if negative
    //      if (negativeCount % negativeStride == 0) {
    //        indices.push_back(it.GetIndex());
    //      }
    //      negativeCount++;
    //    }
    //    else {
    //      indices.push_back(it.GetIndex());
    //    }
    //  }
    //} else
    {
      std::cout << "use mask" << std::endl;
      itk::ImageRegionConstIterator<BinaryImage3D> itMask(mask, shrinkRegion);

      for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
        if (itMask.Get() != 0) {
          auto& index = itMask.GetIndex();

          if (label2->GetPixel(index) != 0) {// if 2 class
            indices.push_back(index);
            class2Count++;
          } else if (label1->GetPixel(index) != 0) {// if 1 class
            indices.push_back(index);
            class1Count++;
          } else {
            if (negativeCount % negativeStride == 0) {
              indices.push_back(index);
            }
            negativeCount++;
          }
        }
      }
    }
    const int totalCount = indices.size();
    negativeCount /= negativeStride;

    std::cout << "class1Count: " << class1Count << std::endl;
    std::cout << "class2Count: " << class2Count << std::endl;
    std::cout << "negativeCount: " << negativeCount << std::endl;
    std::cout << "totalCount: " << totalCount << std::endl;

    // set up output directories
    std::string iImageStr = inputDir.substr(inputDir.length() - 3);//take 3 last chars, e.g. 012
    std::string outDir = outputFolder + "\\" + iImageStr + "\\";
    system((std::string("md ") + outDir).c_str());

    if (is4Classes) {
      system((std::string("md ") + outDir + TP).c_str());
      system((std::string("md ") + outDir + TN).c_str());
      system((std::string("md ") + outDir + FP).c_str());
      system((std::string("md ") + outDir + FN).c_str());
    }
    else { //if 3 classes
      system((std::string("md ") + outDir + "0").c_str());
      system((std::string("md ") + outDir + "1").c_str());
      system((std::string("md ") + outDir + "2").c_str());

    }

    for (int j = 0; j < totalCount; ++j) {
      auto& index = indices[j];
      auto tile = getTile<UInt8Pixel>(image, index, radius);

      //save image
      std::string indexStr = std::to_string(index[0]) + "_" + std::to_string(index[1]) + "_" + std::to_string(index[2]);
      auto label1I = label1->GetPixel(index);
      auto label2I = label2->GetPixel(index);
      std::string labelStr;
      // boosted classes will not be implemented
      //if (is4Classes) {
      //  auto adaI = adaptive->GetPixel(index);
      //  if (labelI == 1 && adaI == 1) {
      //    labelStr = TP;
      //  }
      //  else if (labelI == 0 && adaI == 0) {
      //    labelStr = TN;

      //  }
      //  else if (labelI == 0 && adaI == 1) {
      //    labelStr = FP;

      //  }
      //  else {
      //    labelStr = FN;

      //  }
      //} else 
      { // 2 classes
        labelStr = label2I != 0 ? '2' : (label1I != 0 ? '1' : '0');
      }
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
