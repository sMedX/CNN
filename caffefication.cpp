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
  // perfroms classifying of i-th batch (batchLength items of indices) of input image and store result in classifiedImage
  // image and classifiedImage must have same geometry
  // classifiedImage must be initializated
  // return 
  void classifyIthBatch(caffe::Net<float>* caffeNet, const FloatImage3D::Pointer& image, int radius, int batchLength,
    int groupX, int groupY, int classCount, int channels, std::vector<Image3DIndex>& indices,
    itk::Image<unsigned char, 3>::Pointer classifiedImage, int& posCount, int i)
  {
    auto time0 = clock();

    float* buffer = image->GetBufferPointer();

    // tile's properties
    const int height = 2 * radius;
    const int width = 2 * radius;
    const int tileSize = width*height*channels;
    const int lineSizeInBytes = 2 * radius*sizeof(float);

    // image's properties
    const auto& imageSize = image->GetLargestPossibleRegion().GetSize();
    const int sliceSize = imageSize[0] * imageSize[1];
    const int lineSize = imageSize[1];


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

      if (channels == 3) {
        for (int j = -1; j < 2; j++) { // -1, 0, 1
          const int zOffset = (index[2] + j) * sliceSize;
          const float* src = buffer + zOffset + yOffsetPart + xOffset;

          for (int iRow = 0; iRow < 2 * radius; iRow++) {
            memcpy(dst, src, lineSizeInBytes);

            src += lineSize; // adjust yOffset
            dst += width; // adjust lineOffset
          }
        }

      } else {
        const int zOffset = index[2] * sliceSize;
        const float* src = buffer + zOffset + yOffsetPart + xOffset;

        for (int iRow = 0; iRow < 2 * radius; iRow++) { // TODO try to compute offsets by 1 vector command
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

    }

    std::cout << "load data: " << static_cast<double>(time1 - time0) / CLOCKS_PER_SEC << std::endl;
    std::cout << "classify: " << static_cast<double>(time2 - time1) / CLOCKS_PER_SEC << std::endl;
  }

  bool classify(caffe::Net<float>* caffeNet, const std::string& preset, Int16Image3D::Pointer image16,
  UInt8Image3D::Pointer imageMask, Image3DRegion& region, int radius, float spacingXY, int batchLength, int groupX,
  int groupY, int classCount, bool isRgb, BinaryImage3D::Pointer& outImage)
{
  if (classCount < 2 && classCount > 3) {
    std::cout << "classCount must be 2, 3 or 4";
  }

  int channels = isRgb ? 3 : 1;

  std::cout << "isRgb: " << isRgb << std::endl;
  std::cout << "channels: " << channels << std::endl;

  Image3DSpacing spacing;
  spacing[0] = spacingXY;
  spacing[1] = spacingXY;
  spacing[2] = 0;
  UInt8Image3D::Pointer image8;
  {
    UInt8Image3D::Pointer imgage8Tmp = smartCastImage(preset, image16, imageMask);
    image8 = preprocess(radius, spacing, isRgb, imgage8Tmp);
  }
  imageMask = preprocessBinary(radius, spacing, isRgb, imageMask);

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
    radius3D = { radius, radius, 0 };
  } else {
    radius3D = { radius, radius, 1 };
  }
  shrinkRegion.ShrinkByRadius(radius3D);
  region.Crop(shrinkRegion);

  std::vector<Image3DIndex> indices;
  std::cout << "region: " << region << std::endl;
  if (imageMask.IsNotNull()) {
    std::cout << "use mask" << std::endl;
    // todo there is mistake with region calculation. It's noticable on very littly masks but noticable.
    itk::ImageRegionConstIterator<BinaryImage3D> itMask(imageMask, imageMask->GetLargestPossibleRegion()/*region*/);

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
    std::cout << "WARNING: region calculation may be not precise, use mask instead" << std::endl;
    itk::ImageRegionConstIterator<FloatImage3D> it(image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
      //take central pixel of group
      auto& index = it.GetIndex();
      if (index[0] % groupX == groupX / 2 && index[1] % groupY == groupY / 2) {
        indices.push_back(it.GetIndex());
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

  if (totalCount == 0) {
    std::cout << "error: empty area" << std::endl;
    return false;
  }

  auto classifiedImage = BinaryImage3D::New();
  classifiedImage->CopyInformation(image);
  classifiedImage->SetRegions(image->GetLargestPossibleRegion());
  classifiedImage->Allocate();
  classifiedImage->FillBuffer(0);

  std::cout << "." << std::endl;

  //reshape the net to fill last batch
  const int height = 2 * radius, width = 2 * radius;;
  caffeNet->blob_by_name("data")->Reshape(batchLength, channels, height, width);
  caffeNet->Reshape(); // optional -- the net will reshape automatically before a call to forward()

  auto newShape = caffeNet->input_blobs()[0]->shape();
  std::cout << "new shape: " << newShape[0] << ", " << newShape[1] << ", " << newShape[2] << ", " << newShape[3] << std::endl;

  int posCount = 0;

  std::cout << "Applying CNN in deploy config" << std::endl;

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);

  auto time0 = clock();

  for (int i = 0; i < totalCount / batchLength; ++i) {
    classifyIthBatch(caffeNet, image, radius, batchLength, groupX, groupY, classCount, channels, indices, classifiedImage, posCount, i);

    if ((i+1) % 10 == 0) {
      std::cout << i + 1 << " of  " << totalCount / batchLength + 1 << std::endl;
    }
  }
  auto time1 = clock();

  std::cout << "rest batch" << std::endl;

  if (totalCount % batchLength != 0) {
    // precess last not full batch
    int batchLengthRest = totalCount % batchLength;
    int iRest = totalCount / batchLength;
    int posCountRest = 0;

    std::cout << " and rest: "<< iRest + 1 << " of  " << totalCount / batchLength + 1 << std::endl;
    std::cout << "batchLengthRest: " << batchLengthRest << std::endl;

    //reshape the net to fill last batch
    caffeNet->blob_by_name("data")->Reshape(batchLengthRest, channels, height, width);
    caffeNet->Reshape(); // optional -- the net will reshape automatically before a call to forward()

    auto newShape = caffeNet->input_blobs()[0]->shape();
    std::cout << "new shape: " << newShape[0] << ", " << newShape[1] << ", " << newShape[2] << ", " << newShape[3] << std::endl;

    classifyIthBatch(caffeNet, image, radius, batchLengthRest, groupX, groupY, classCount, channels, indices, classifiedImage, posCountRest, iRest);

    posCount += posCountRest;
  }
  auto time2 = clock();

  if (totalCount / batchLength > 1){
    std::cout << "performance for full-batched parts" << std::endl;
    double time = (time1 - time0) / CLOCKS_PER_SEC;
    auto count = (totalCount / batchLength)*batchLength;
    auto avgTime = time / count;

    std::cout << "avgerage time (ms) per 1000 units: " << avgTime * 1000 << std::endl;;
  } else {
    std::cout << "performance for part-batched part" << std::endl;
    double time = (time2 - time1) / CLOCKS_PER_SEC;
    auto count = totalCount % batchLength;
    auto avgTime = time / count;

    std::cout << "avgerage time (ms) per 1000 units: " << avgTime*1000 << std::endl;
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
  if (deviceId == -1) {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(deviceId);       
  }

  std::cout << "create net" << std::endl;
  caffeNet = std::make_shared<caffe::Net<float>>(modelFile, caffe::TEST);
  std::cout << "load net's weights" << std::endl;
  caffeNet->CopyTrainedLayersFrom(trainedFile);
}
}
