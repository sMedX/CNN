// caffefication.cpp : Defines the exported functions for the DLL.

//#include <cstring>
//#include <cstdlib>
#include <iostream>

// ITK
#include <itkImage.h>

#include <caffe/caffe.hpp>
#include <caffe/blob.hpp>

#include <agtkResampling.h>

#include "preprocess.h"
#include "caffefication.h"

namespace caffefication {
bool classify(caffe::Net<float>* caffeNet, const std::string& preset, Int16Image3D::Pointer image16,
  UInt8Image3D::Pointer imageMask, Image3DRegion& region, int radiusXY, float spacingXY, int batchLength, int groupX,
  int groupY, int classCount, bool isRgb, OUT BinaryImage3D::Pointer& outImage)
{
  if (classCount < 1 && classCount > 3) {
    std::cout << "classCount must be 1, 2, 3 or 4";
  }

  UInt8Image3D::Pointer image8 = UInt8Image3D::New();
  UInt8Image3D::Pointer imageNull = nullptr;

  preprocess(radiusXY, preset, spacingXY, isRgb, image16, imageNull, imageNull, imageMask, imageNull, image8);

  std::cout << "cast image to float" << std::endl;
  typedef itk::CastImageFilter<UInt8Image3D, FloatImage3D> Cast;
  auto cast = Cast::New();
  cast->SetInput(image8);
  cast->Update();
  FloatImage3D::Pointer image = cast->GetOutput();
  if (imageMask != nullptr) {
    if (image->GetLargestPossibleRegion() != imageMask->GetLargestPossibleRegion()) {
      std::cout << "image->GetLargestPossibleRegion() != imageMask->GetLargestPossibleRegion() " << std::endl;
      return false;
    }
  }

  std::cout << "Calculating indices" << std::endl;
  std::cout << "region in original spacing " << region << std::endl;

  for (size_t i = 0; i < IMAGE_DIM_2; ++i) {
    region.GetModifiableIndex()[i] *= spacingXY;
    region.GetModifiableSize()[i] *= spacingXY;
  }
  std::cout << "region in modified spacing " << region << std::endl;

  auto shrinkRegion = image->GetLargestPossibleRegion();
  Image3DSize radius3D;
  if (isRgb) {
    radius3D = { radiusXY, radiusXY, 0 };
  } else {
    radius3D = { radiusXY, radiusXY, 1 };
  }
  shrinkRegion.ShrinkByRadius(radius3D);
  region.Crop(shrinkRegion);

  std::vector<Image3DIndex> indices;
  std::cout << "region: " << region << std::endl;
  if (imageMask.IsNotNull()) {
    std::cout << "use mask" << std::endl;
    itk::ImageRegionConstIterator<BinaryImage3D> itMask(imageMask, region);

    for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
      if (itMask.Get() != 0) {
        //take central pixel of group
        const auto& index = itMask.GetIndex();
        if (index[0] % groupX == groupX / 2 && index[1] % groupY == groupY / 2) {
          indices.push_back(itMask.GetIndex());
        }
      }
    }
  } else {
    std::cout << "not use mask" << std::endl;
    itk::ImageRegionConstIterator<FloatImage3D> itMask(image, region);
    for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
      //take central pixel of group
      auto& index = itMask.GetIndex();
      if (index[0] % groupX == groupX / 2 && index[1] % groupY == groupY / 2) {
        indices.push_back(itMask.GetIndex());
      }
    }
  }

  //reorder like 'zxy'
  //std::stable_sort(indices.begin(), indices.end(), [](agtk::Image3DIndex a, agtk::Image3DIndex b) // todo maybe use ordinal sort
  //{
  //  return a[2] != b[2] ? a[2] < b[2] : a[0] < b[0];
  //});
  const int totalCount = indices.size();

  std::cout << "total count:" << totalCount << std::endl;

  auto classifiedImage = BinaryImage3D::New();
  classifiedImage->CopyInformation(image);
  classifiedImage->SetRegions(image->GetLargestPossibleRegion());
  classifiedImage->Allocate();
  classifiedImage->FillBuffer(0);

  std::cout << "." << std::endl;

  int itCount = 0;
  int posCount = 0;

  std::cout << "Applying CNN in deploy config" << std::endl;

  float* buffer = image->GetBufferPointer();

  int channels = isRgb ? 3 : 1;

  std::cout << "isRgb: " << isRgb << std::endl;
  std::cout << "channels: " << channels << std::endl;

  // tile's properties
  const int height = 2 * radiusXY;
  const int width = 2 * radiusXY;
  const int tileSize = width*height*channels;
  const int lineSizeInBytes = 2 * radiusXY*sizeof(float);

  // image's properties
  const auto& imageSize = image->GetLargestPossibleRegion().GetSize();
  const int sliceSize = imageSize[0] * imageSize[1];
  const int lineSize = imageSize[1];

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);

  for (int i = 0; i < totalCount / batchLength; ++i) { //todo fix last total%batch_size indices
    auto time0 = clock();
    std::cout << i << "th batch" << std::endl;

    caffe::Blob<float>* blob = new caffe::Blob<float>(batchLength, channels, height, width);

    //int lastX = -1; // we dont use last z and assume thay 'z' stay same
    for (int iTile = 0; iTile < batchLength; ++iTile) {
      const auto& index = indices[i*batchLength + iTile];
      //if (index[0] == lastX) { // x same as before
      //concat new line(s) to last tile
      //} else {
      //make tile from the scratch
      const int xOffset = (index[0]/* - radiusXY*/ + 1); //TODO mb adjust index of image?
      const int yOffsetPart = (index[1]/* - radiusXY*/ + 1)* lineSize;

      const int tileOffset = iTile*tileSize; // todo remove these line dst buffer is adjusted by +=
      float* dst = blob->mutable_cpu_data() + tileOffset;

      if (isRgb) {
        for (int j = -1; j < 2; j++) { // -1, 0, 1
          const int zOffset = (index[2] + j) * sliceSize;
          const float* src = buffer + zOffset + yOffsetPart + xOffset;

          for (int iRow = 0; iRow < 2 * radiusXY; iRow++) {
            memcpy(dst, src, lineSizeInBytes);

            src += lineSize; // adjust yOffset
            dst += width; // adjust lineOffset
          }
        }

      } else {
        const int zOffset = index[2] * sliceSize;
        const float* src = buffer + zOffset + yOffsetPart + xOffset;

        for (int iRow = 0; iRow < 2 * radiusXY; iRow++) { // try to compute offset by 1 vector command
          memcpy(dst, src, lineSizeInBytes);

          src += lineSize; // adjust yOffset
          dst += width; // adjust lineOffset
        }
      }
    }

    //fill the vector
    std::vector<caffe::Blob<float>*> bottom;
    bottom.push_back(blob);
    float type = 0.0;

    auto time1 = clock();

    auto results = caffeNet->Forward(bottom, &type)[0]->cpu_data();
    delete blob;

    auto time2 = clock();
    //Here I can use the argmax layer, but for now I do a simple for :)
    for (int iTile = 0; iTile < batchLength; ++iTile) {
      auto& index = indices[i*batchLength + iTile];

      int val = results[iTile];

      if (classCount == 4) {
        const int TP = 2, FN = 3; // there are labels from last classificatoin onto 2 classes

        if (val == TP || val == FN) { // TP,FN -> true, TN,FP ->false
          val = 1;
          posCount++;
        } else {
          val = 0;
        }
      } else {
        if (val != 0) {
          posCount++;
          val = 1;
        }
      } 
      //set group's area
      for (int k = -groupX / 2; k < groupX - groupX / 2; ++k) {
        for (int l = -groupY / 2; l < groupY - groupY / 2; ++l) {
          Image3DSize offset = { k, l, 0 };
          auto index2 = index + offset;
          classifiedImage->SetPixel(index2, val); // can be improved if only group 1x1 used
        }
      }

      //
      if (++itCount % 10000 == 0) {
        std::cout << itCount << " / " << totalCount << "\n";
      }
    }

    std::cout << "load data: " << static_cast<double>(time1 - time0) / CLOCKS_PER_SEC << std::endl;
    std::cout << "classify: " << static_cast<double>(time2 - time1) / CLOCKS_PER_SEC << std::endl;

  }

  std::cout << "positives:" << posCount << std::endl;

  //postprocess
  //resampling back
  if (spacingXY != 0) { //resample image by axial slices
    std::cout << "resample" << std::endl;
    typedef itk::ResampleImageFilter<BinaryImage3D, BinaryImage3D> ResampleImageFilterType;

    auto nn_interpolator = itk::NearestNeighborInterpolateImageFunction<BinaryImage3D>::New();

    auto resampleFilter = ResampleImageFilterType::New();
    resampleFilter->SetInput(classifiedImage);
    resampleFilter->SetReferenceImage(image16);
    resampleFilter->SetUseReferenceImage(true);
    resampleFilter->SetInterpolator(nn_interpolator);
    resampleFilter->SetDefaultPixelValue(0);
    resampleFilter->UpdateLargestPossibleRegion();
    outImage = resampleFilter->GetOutput();
  }
  return true;
}

void loadNet(const std::string& modelFile, const std::string& trainedFile, int deviceId, OUT std::shared_ptr<caffe::Net<float>>& caffeNet)
{
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(deviceId);

  std::cout << "create net" << std::endl;
  caffeNet = std::make_shared<caffe::Net<float>>(modelFile, caffe::TEST);
  std::cout << "load net's weights" << std::endl;
  caffeNet->CopyTrainedLayersFrom(trainedFile);
}
}
