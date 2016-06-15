///
/// Cuts image by tiles (by the corresponded mask) and saves it in folders positive/ and negative/. Adds file list text file.
/// If mask provided then it used as negative. Else use bounding box as negative.
///
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>

#include <itkImage.h>
#include <itkTestingExtractSliceImageFilter.h>
#include <itkMetaImageIOFactory.h>
#include <itkNrrdImageIOFactory.h>

#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkCommandLineArgumentParser.h"

#include "preprocess.h"

namespace caffefication {

agtk::BinaryImage2D::Pointer getTile(const agtk::BinaryImage3D* image, const itk::ImageBase<3>::IndexType& index, int halfSize)
{
  typedef agtk::BinaryImage3D ImageType3D;
  typedef agtk::BinaryImage2D ImageType2D;

  agtk::Image2DSize size2D = { 2 * halfSize, 2 * halfSize };

  auto ret = ImageType2D::New();
  ret->SetRegions(size2D);
  ret->Allocate();

  agtk::Image3DSize size = { 2 * halfSize, 2 * halfSize, 0 };
  agtk::Image3DIndex start = { index[0] - halfSize + 1, index[1] - halfSize + 1, index[2] };

  agtk::Image3DRegion region;
  region.SetSize(size);
  region.SetIndex(start);

  itk::ImageRegionIterator<ImageType2D> itOut(ret, ret->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<ImageType3D> itIn(image, region);

  for (itOut.GoToBegin(), itIn.GoToBegin(); !itOut.IsAtEnd(); ++itOut, ++itIn) {
    itOut.Set(itIn.Value());
  }

  return ret;
}

itk::Image<itk::RGBPixel<UINT8>, 2>::Pointer getRGBTile(const agtk::BinaryImage3D* image, const itk::ImageBase<3>::IndexType& index, int halfSize)
{
  typedef agtk::BinaryImage3D ImageType3D;
  typedef itk::Image<itk::RGBPixel<UINT8>, 2> ImageType2D;

  agtk::Image2DSize size2D = { 2 * halfSize, 2 * halfSize };

  auto ret = ImageType2D::New();
  ret->SetRegions(size2D);
  ret->Allocate();

  agtk::Image3DSize size = { 2 * halfSize, 2 * halfSize, 0 };
  agtk::Image3DIndex start = { index[0] - halfSize + 1, index[1] - halfSize + 1, index[2] };
  agtk::Image3DIndex startUp = { index[0] - halfSize + 1, index[1] - halfSize + 1, index[2] + 1 };
  agtk::Image3DIndex startDown = { index[0] - halfSize + 1, index[1] - halfSize + 1, index[2] - 1 };

  agtk::Image3DRegion region;
  region.SetSize(size);
  region.SetIndex(start);

  agtk::Image3DRegion regionUp;
  regionUp.SetSize(size);
  regionUp.SetIndex(startUp);

  agtk::Image3DRegion regionDown;
  regionDown.SetSize(size);
  regionDown.SetIndex(startDown);

  itk::ImageRegionIterator<ImageType2D> itOut(ret, ret->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<ImageType3D> itIn(image, region);
  itk::ImageRegionConstIterator<ImageType3D> itInUp(image, regionUp);
  itk::ImageRegionConstIterator<ImageType3D> itInDown(image, regionDown);

  for (itOut.GoToBegin(), itIn.GoToBegin(), itInUp.GoToBegin(), itInDown.GoToBegin();
    !itOut.IsAtEnd();
    ++itOut, ++itIn, ++itInUp, ++itInDown) {
    ImageType2D::PixelType val;
    val[0] = itInDown.Value();
    val[1] = itIn.Value();
    val[2] = itInUp.Value();
    itOut.Set(val);
  }

  return ret;
}
}

int main(int argc, char* argv[])
{
  using namespace agtk;
  using namespace caffefication;

  const std::string ext = ".png";
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

  Image3DSize stride = { 1, 1, 1 }; //default no-stride
  parser->GetITKValue<Image3DSize>("-stride", stride);

  std::string preset;
  parser->GetValue("-preset", preset);

  float spacingXY;
  parser->GetValue("-spacingXY", spacingXY);

  int strideNegative = 4; // additional stride for negative points
  parser->GetValue("-strideNegative", strideNegative);

  bool isRgb = false;
  parser->GetValue("-rgb", isRgb);

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
  //std::cout << "spacingXY " << spacingXY << std::endl;
  std::cout << "stride for negative points is " << strideNegative << std::endl;
  std::cout << "is Rgb " << isRgb << std::endl;

  bool isNoMask = false;
  if (maskName == NO_MASK) {
    isNoMask = true;
  }
  std::vector<std::string> inputDirs;

  std::ifstream infile(listFile);
  std::string line;

  while (std::getline(infile, line)) {
    inputDirs.push_back(line);
  }
  std::cout << "inputData.size() " << inputDirs.size() << std::endl;

  // set up output directories
  std::cout << outputFolder << std::endl;
  system((std::string("md ") + outputFolder).c_str());

  //itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
  //omp_set_num_threads(24); // Use 24 threads

  //int c = 0;

#pragma omp parallel for
  for (int iImage = 0; iImage < inputDirs.size(); ++iImage) {
    auto inputDir = inputDirs[iImage];

    bool isAdaptiveClasses = !adaptiveName.empty();
    bool isLabel2 = !labelName2.empty();

    // read images
    std::cout << "load image" << std::endl;
    auto imageFile = inputDir + "\\" + imageName;
    Int16Image3D::Pointer image16 = Int16Image3D::New();
    if (!readImage<Int16Image3D>(image16, imageFile)) {
      std::cout << "can't read " << imageFile;
      continue;
    }

    std::cout << "load label1" << std::endl;
    auto labelFile1 = inputDir + "\\" + labelName1;
    std::cout << "labelFile1 " << labelFile1 << std::endl;
    BinaryImage3D::Pointer label1 = BinaryImage3D::New();
    if (!readImage(label1, labelFile1)) {
      std::cout << "can't read " << labelFile1;
      continue;
    }

    std::cout << "load label2" << std::endl;
    BinaryImage3D::Pointer label2 = BinaryImage3D::New();
    if (isLabel2) {
      auto labelFile2 = inputDir + "\\" + labelName2;
      std::cout << "labelFile2 " << labelFile2 << std::endl;
      if (!readImage(label2, labelFile2)) {
        std::cout << "can't read " << labelFile2 << ". label2 will not be used." << std::endl;
        label2->CopyInformation(label1); // make an empty image instead
        label2->Allocate();
        label2->FillBuffer(0);
      }
    } else {
      label2 = nullptr;
    }

    BinaryImage3D::Pointer mask = BinaryImage3D::New();
    if (!isNoMask) {
      std::cout << "load mask" << std::endl;
      auto maskFile = inputDir + "\\" + maskName;
      std::cout << "maskFile " << maskFile << std::endl;
      if (!readImage(mask, maskFile)) {
        std::cout << "can't read " << maskFile << std::endl;
        continue;
      }
    } else {
      mask->SetRegions(image16->GetLargestPossibleRegion());
      mask->Allocate();
      mask->FillBuffer(1);
    }

    BinaryImage3D::Pointer adaptive = BinaryImage3D::New();
    if (isAdaptiveClasses) {
      std::cout << "load adaptive" << std::endl;
      auto adaptiveFile = inputDir + "\\" + adaptiveName; 
      std::cout << "adaptiveFile " << adaptiveFile << std::endl;
      if (!readImage(adaptive, adaptiveFile)) {
        std::cout << "can't read " << adaptiveFile << std::endl;
        continue;
      }
    } else {
      adaptive = nullptr;
    }

    //todo add check for equal size

    // organ-based transformation to UINT8 from int16
    auto image = smartCastImage(preset, image16, mask);

    //hardcoded consts
    std::vector<float> spacingXYVector = {spacingXY};
    for (float spacingXY : spacingXYVector) {
      std::cout << "preprocess images" << std::endl;
      std::cout << "spacing :" << spacingXY << std::endl;

      UInt8Image3D::Pointer imagePreproc = preprocess(radius, spacingXY, isRgb, image),
        label1Preproc = preprocessBinary(radius, spacingXY, isRgb, label1), // todo use binaryResampling
        label2Preproc = label2 ? preprocessBinary(radius, spacingXY, isRgb, label2) : nullptr,
        maskPreproc = mask ? preprocessBinary(radius, spacingXY, isRgb, mask) : nullptr,
        adaptivePreproc = adaptive ? preprocessBinary(radius, spacingXY, isRgb, adaptive) : nullptr;

      auto wholeRegion = imagePreproc->GetLargestPossibleRegion();
      std::cout << "new region: " << wholeRegion << std::endl;

      std::cout << "calculate indices" << std::endl;
      std::vector<BinaryImage3D::IndexType> indices;

      int negativeCount = 0;
      int class1Count = 0;
      int class2Count = 0;

      itk::ImageRegionConstIterator<BinaryImage3D> itMask(maskPreproc, wholeRegion);

      for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
        if (itMask.Get() != 0) {
          auto& index = itMask.GetIndex();
          if (index[0] % stride[0] == 0 && index[1] % stride[1] == 0 && index[2] % stride[2] == 0) { // striding for whole image
            if (label2Preproc.IsNotNull() && label2Preproc->GetPixel(index) != 0) {// if 3 class
              indices.push_back(index);
              class2Count++;
            } else if (label1Preproc->GetPixel(index) != 0) {// if 2 class
              indices.push_back(index);
              class1Count++;
            } else {// if negative
              if (negativeCount % strideNegative == 0) { // striding only for negative
                indices.push_back(index);
              }
              negativeCount++;
            }
          }
        }
      }
      
      const int totalCount = indices.size();
      negativeCount /= strideNegative;

      std::cout << "class1Count: " << class1Count << std::endl;
      std::cout << "class2Count: " << class2Count << std::endl;
      std::cout << "negativeCount: " << negativeCount << std::endl;
      std::cout << "totalCount: " << totalCount << std::endl;

      // set up output directories
      std::string iImageStr = inputDir.substr(inputDir.length() - 3);//take 3 last chars, e.g. 012
      std::string outDir = outputFolder + "\\" + iImageStr + "\\";
      system((std::string("md ") + outDir).c_str());

      if (isAdaptiveClasses) {
        system((std::string("md ") + outDir + TP).c_str());
        system((std::string("md ") + outDir + TN).c_str());
        system((std::string("md ") + outDir + FP).c_str());
        system((std::string("md ") + outDir + FN).c_str());
      } else { //if 2 or 3 classes
        system((std::string("md ") + outDir + "0").c_str());
        system((std::string("md ") + outDir + "1").c_str());
        if (class2Count != 0) {
          system((std::string("md ") + outDir + "2").c_str());
        }
      }

      for (int j = 0; j < totalCount; ++j) {
        auto& index = indices[j];

        //name image
        std::string indexStr = std::to_string(index[0]) + "_" + std::to_string(index[1]) + "_" + std::to_string(index[2]);
        auto label1I = label1Preproc->GetPixel(index); // TODO use list of indices a-priori. don't read label image at this point
        std::string labelStr;
        // boosted classes
        if (isAdaptiveClasses) {
          auto adaI = adaptivePreproc->GetPixel(index);
          if (label1I == 1 && adaI == 1) {
            labelStr = TP;
          } else if (label1I == 0 && adaI == 0) {
            labelStr = TN;
          } else if (label1I == 0 && adaI == 1) {
            labelStr = FP;
          } else {
            labelStr = FN;
          }
        } else { // 2 or 3 classes
          if (class2Count != 0) { // 3 class
            auto label2I = label2Preproc->GetPixel(index);
            labelStr = label2I != 0 ? '2' : (label1I != 0 ? '1' : '0');
          } else {
            labelStr = label1I != 0 ? '1' : '0';
          }
        }
        std::string filename = outDir + labelStr + "/" + indexStr + +"-s" + std::to_string(spacingXY) + ext;

        if (isRgb) {
          writeImage(getRGBTile(imagePreproc, index, radius).GetPointer(), filename);
        } else {
          writeImage(getTile(imagePreproc, index, radius).GetPointer(), filename);
        }

        if (j % 10000 == 0) {
          std::cout << static_cast<double>(j * 100) / totalCount << "% of " << iImageStr << " image" << std::endl;
        }
      }
    }
  }
  return EXIT_SUCCESS;
};
