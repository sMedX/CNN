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
#include <itkConstantPadImageFilter.h>

#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkCommandLineArgumentParser.h"
#include "agtkBinaryImageUtilities.h"
#include "agtkResampling.h"

agtk::BinaryImage2D::Pointer getTile(const agtk::BinaryImage3D* image, const itk::ImageBase<3>::IndexType& index, int halfSize);

itk::Image<itk::RGBPixel<UINT8>, 2>::Pointer getRGBTile(const agtk::BinaryImage3D* image, const itk::ImageBase<3>::IndexType& index, int halfSize);

agtk::BinaryImage3D::Pointer padImage(const agtk::BinaryImage3D* image, const itk::ImageBase<3>::SizeType& outputRegion);

int main(int argc, char* argv[])
{
  using namespace agtk;

  const std::string ext = ".png";
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
  std::cout << "spacingXY " << spacingXY << std::endl;
  std::cout << "stride for negative points is " << strideNegative << std::endl;
  std::cout << "is Rgb " << isRgb << std::endl;

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

    bool isAdaptiveClasses = !adaptiveName.empty();

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
    if (isAdaptiveClasses) {
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
    } else if (preset == "livertumors") {
      const int shift = 40;

      // x' = x + shift
      itk::ImageRegionIterator<Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(it.Get() + shift);
      }
    }
    std::cout << "cast (truncate)" << std::endl;
    // force integer overflow
    UInt8Image3D::PixelType minValue = 0, maxValue = 255;

    itk::ImageRegionIterator<Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
      auto val = it.Get();
      if (val < minValue) {
        val = minValue;
      } else if (val > maxValue) {
        val = maxValue;
      }
      it.Set(val);
    }

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
      if (label2.IsNotNull()) {
        label2 = resamplingBinary(label2.GetPointer(), spacing);
      }

      if (mask.IsNotNull()) {
        mask = resamplingBinary(mask.GetPointer(), spacing);
      }

      if (isAdaptiveClasses) {
        adaptive = resamplingBinary(adaptive.GetPointer(), spacing);
      }
    }

    const Image3DOffset radius3D = { radius, radius, isRgb ? 1 : 0 };
    const Image3DSize size3D = { radius3D[0], radius3D[1], radius3D[2] };
    std::cout << "padding by radius " << size3D << std::endl;

    image = padImage(image, size3D);
    label1 = padImage(label1.GetPointer(), size3D);
    if (label2.IsNotNull()) {
      label2 = padImage(label2.GetPointer(), size3D);
    }

    if (mask.IsNotNull()) {
      mask = padImage(mask.GetPointer(), size3D);
    }

    if (isAdaptiveClasses) {
      adaptive = padImage(adaptive.GetPointer(), size3D);
    }
    auto wholeRegion = image->GetLargestPossibleRegion();
    std::cout << "new region: " << wholeRegion << std::endl;

    std::cout << "calculate indices" << std::endl;
    std::vector<BinaryImage3D::IndexType> indices;

    int negativeCount = 0;
    int class1Count = 0;
    int class2Count = 0;

    if (isBoundingBox) { // not take in count 3-class problem
      auto region = getBinaryMaskBoundingBoxRegion(label1);
      region.Crop(wholeRegion);
      std::cout << "use bounding box: " << region << std::endl;

      itk::ImageRegionConstIterator<BinaryImage3D> itlabel(label1, region);

      for (itlabel.GoToBegin(); !itlabel.IsAtEnd(); ++itlabel) {
        if (label1->GetPixel(itlabel.GetIndex()) == 0) {// if negative
          if (negativeCount % strideNegative == 0) {
            indices.push_back(itlabel.GetIndex());
          }
          negativeCount++;
        } else {
          indices.push_back(itlabel.GetIndex());
        }
      }
      class1Count = indices.size() - negativeCount / strideNegative;
    } else if (isNoMask) { // not take in count 3-class problem
      // this part can be removed when ordinal white image will be used as mask
      std::cout << "don't mask" << std::endl;
      itk::ImageRegionConstIterator<BinaryImage3D> itLabel(label1, wholeRegion);
      for (itLabel.GoToBegin(); !itLabel.IsAtEnd(); ++itLabel) {
        if (label1->GetPixel(itLabel.GetIndex()) == 0) {// if negative
          if (negativeCount % strideNegative == 0) {
            indices.push_back(itLabel.GetIndex());
          }
          negativeCount++;
        } else {
          indices.push_back(itLabel.GetIndex());
        }
      }
      class1Count = indices.size() - negativeCount / strideNegative;
    } else {
      std::cout << "use mask" << std::endl;
      itk::ImageRegionConstIterator<BinaryImage3D> itMask(mask, wholeRegion);

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
            if (negativeCount % strideNegative == 0) {
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
      auto label1I = label1->GetPixel(index);
      std::string labelStr;
      // boosted classes
      if (isAdaptiveClasses) {
        auto adaI = adaptive->GetPixel(index);
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
          auto label2I = label2->GetPixel(index);
          labelStr = label2I != 0 ? '2' : (label1I != 0 ? '1' : '0');
        } else {
          labelStr = label1I != 0 ? '1' : '0';
        }
      }
      std::string filename = outDir + labelStr + "\\" + indexStr + ext;

      if (isRgb) {
        writeImage(getRGBTile(image, index, radius).GetPointer(), filename);
      } else {
        writeImage(getTile(image, index, radius).GetPointer(), filename);
      }

      if (j % 10000 == 0) {
        std::cout << static_cast<double>(j * 100) / totalCount << "% of " << iImageStr << " image" << std::endl;
      }
    }
  }
  return EXIT_SUCCESS;
};

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

agtk::BinaryImage3D::Pointer padImage(const agtk::BinaryImage3D* image, const itk::ImageBase<3>::SizeType& outputRegion)
{
  typedef agtk::BinaryImage3D ImageType;
  typedef itk::ConstantPadImageFilter <ImageType, ImageType> ConstantPadImageFilterType;

  const ImageType::PixelType constantPixel = 0;

  auto padFilter = ConstantPadImageFilterType::New();
  padFilter->SetInput(image);
  padFilter->SetPadBound(outputRegion); // Calls SetPadLowerBound(region) and SetPadUpperBound(region)
  padFilter->SetConstant(constantPixel);
  padFilter->Update();

  return padFilter->GetOutput();
}