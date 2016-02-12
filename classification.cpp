
#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>

// ITK
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMetaImageIOFactory.h>
#include <itkNrrdImageIOFactory.h>

#include "C:/alex/agtk/Modules/Core/agtkResampling.h" //todo remove it or intergrate agtk

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"

int main(int argc, char** argv)
{
  using namespace caffe;

  string model_file = argv[1];
  string trained_file = argv[2];

  string start_x_str = argv[3];
  string start_y_str = argv[4];
  string start_z_str = argv[5];

  string size_x_str = argv[6];
  string size_y_str = argv[7];
  string size_z_str = argv[8];

  string radiusXY_str = argv[9];
  string preset = argv[10];
  string spacingXY_str = argv[11];

  string batchLengthStr = argv[12];

  string groupXStr = argv[13]; // interpret an area XxY as 1 unit
  string groupYStr = argv[14];

  string classCountStr = argv[15];

  string input_file = argv[16];
  string mask_file = argv[17];
  string output_file = argv[18];

  string deviceIdStr = argv[19];

  agtk::Image3DIndex start;
  start[0] = atoi(start_x_str.c_str());
  start[1] = atoi(start_y_str.c_str());
  start[2] = atoi(start_z_str.c_str());

  agtk::Image3DSize size;
  size[0] = atoi(size_x_str.c_str());
  size[1] = atoi(size_y_str.c_str());
  size[2] = atoi(size_z_str.c_str());

  agtk::Image3DRegion region;
  region.SetIndex(start);
  region.SetSize(size);

  int radiusXY = atoi(radiusXY_str.c_str());
  float spacingXY = atof(spacingXY_str.c_str());
  int batchLength = atoi(batchLengthStr.c_str());
  int groupX = atoi(groupXStr.c_str());
  int groupY = atoi(groupYStr.c_str());
  int classCount = atoi(classCountStr.c_str());
  int deviceId = atoi(deviceIdStr.c_str());

  std::cout << "model_file = " << model_file << std::endl <<
    "trained_file =" << trained_file << std::endl <<
    "region = " << region << std::endl <<
    "radiusXY=" << radiusXY << std::endl <<
    "preset=" << preset << std::endl <<
    "spacingXY=" << spacingXY << std::endl <<
    "batchSize=" << batchLength << std::endl <<
    "groupX=" << groupX << std::endl <<
    "groupY=" << groupY << std::endl <<
    "classCount=" << classCount << std::endl <<
    "input_file = " << input_file << std::endl <<
    "mask_file =" << mask_file << std::endl <<
    "output_file =" << output_file << std::endl <<
    "deviceID =" << deviceId << std::endl;

  if (classCount < 1 && classCount > 3) {
    std::cout << "classCount must be 1, 2, 3 or 4";
  }

  std::cout << "load images" << std::endl;

  itk::MetaImageIOFactory::RegisterOneFactory();
  itk::NrrdImageIOFactory::RegisterOneFactory();
  std::cout << "nrrd factory registered" << std::endl;

  typedef std::list<itk::LightObject::Pointer> RegisteredObjectsContainerType;
  RegisteredObjectsContainerType registeredIOs = itk::ObjectFactoryBase::CreateAllInstance("itkImageIOBase");
  std::cout << "there are " << registeredIOs.size() << " IO objects available to the ImageFileReader." << std::endl;

  typedef itk::ImageFileReader<agtk::Int16Image3D> ReaderType;
  typedef itk::ImageFileReader<agtk::BinaryImage3D> BinaryReaderType;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(input_file);
  try {
    reader->Update();
  }
  catch (itk::ExceptionObject &excp) {
    std::cout << "Exception thrown while reading the image " << std::endl;
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "." << std::endl;

  agtk::BinaryImage3D::Pointer imageMask;
  if (mask_file == "BOUNDING_BOX") {
    imageMask = nullptr;
  }
  else {
    BinaryReaderType::Pointer readerMask = BinaryReaderType::New();
    readerMask->SetFileName(mask_file);
    try {
      readerMask->Update();
    }
    catch (itk::ExceptionObject &excp) {
      std::cout << "Exception thrown while reading the mask " << std::endl;
      std::cout << excp << std::endl;
      return EXIT_FAILURE;
    }
    imageMask = readerMask->GetOutput();
  }
  std::cout << "." << std::endl;

  agtk::Int16Image3D::Pointer image16 = reader->GetOutput();

  std::cout << "preprocess images" << std::endl;
  std::cout << "shift, scale images" << std::endl;

  if (preset == "pancreas") {
    const int shift = 190;// b
    const int squeeze = 2;// a

    // x' = (x + b)/a
    itk::ImageRegionIterator<agtk::Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
      it.Set((it.Get() + shift) / squeeze);
    }
  }
  else if (preset == "livertumors") {
    const int shift = 40;

    // x'= x + shift
    itk::ImageRegionIterator<agtk::Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
      it.Set(it.Get() + shift);
    }
  }

  if (spacingXY != 0) {
    std::cout << "resample image's axial slices" << std::endl;

    agtk::Image3DSpacing spacing;
    spacing[0] = spacingXY;
    spacing[1] = spacingXY;
    spacing[2] = image16->GetSpacing()[2];

    image16 = agtk::resampling(image16.GetPointer(), spacing);

    if (imageMask != nullptr) {
      imageMask = agtk::resamplingBinary(imageMask.GetPointer(), spacing);
    }
  }

  std::cout << "cast image to float" << std::endl;
  typedef itk::CastImageFilter<agtk::Int16Image3D, agtk::FloatImage3D> Cast;
  auto cast = Cast::New();
  cast->SetInput(image16);
  cast->Update();
  agtk::FloatImage3D::Pointer image = cast->GetOutput();
  if (imageMask != nullptr) {
    if (image->GetLargestPossibleRegion() != imageMask->GetLargestPossibleRegion()) {
      std::cout << "image->GetLargestPossibleRegion() != imageMask->GetLargestPossibleRegion() " << std::endl;
      return EXIT_FAILURE;
    }
  }
  image16 = nullptr;

  std::cout << "Calculating indices" << std::endl;

  auto shrinkRegion = image->GetLargestPossibleRegion();
  const agtk::Image3DSize radius3D = {radiusXY, radiusXY, 0};
  shrinkRegion.ShrinkByRadius(radius3D);
  region.Crop(shrinkRegion);

  vector<agtk::Image3DIndex> indices;
  std::cout << "region: " << region << std::endl;
  if (imageMask.IsNotNull()) {
    std::cout << "use mask" << std::endl;
    itk::ImageRegionConstIterator<agtk::BinaryImage3D> itMask(imageMask, region);

    for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
      if (itMask.Get() != 0) {
        //take central pixel of group
        const auto& index = itMask.GetIndex();
        if (index[0] % groupX == groupX / 2 && index[1] % groupY == groupY / 2) {
          indices.push_back(itMask.GetIndex());
        }
      }
    }
  }
  else {
    std::cout << "not use mask" << std::endl;
    itk::ImageRegionConstIterator<agtk::FloatImage3D> itMask(image, region);
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

  agtk::BinaryImage3D::Pointer outImage = agtk::BinaryImage3D::New();
  outImage->CopyInformation(image);
  outImage->SetRegions(image->GetLargestPossibleRegion());
  outImage->Allocate();
  outImage->FillBuffer(0);

  std::vector<agtk::BinaryImage3D::Pointer> outImages;
  for (size_t i = 0; i < classCount; i++) {
    outImages.push_back(agtk::BinaryImage3D::New());
    outImages[i]->CopyInformation(image);
    outImages[i]->SetRegions(image->GetLargestPossibleRegion());
    outImages[i]->Allocate();
    outImages[i]->FillBuffer(0);
  }

  std::cout << "." << std::endl;

  int itCount = 0;
  int tumCount = 0;

  std::cout << "Applying CNN in deploy config" << std::endl;

  //Setting CPU or GPU
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(deviceId);

  //get the net
  Net<float> caffe_test_net(model_file, TEST);
  //get trained net
  caffe_test_net.CopyTrainedLayersFrom(trained_file);

  float* buffer = image->GetBufferPointer();

  const int channels = 1;
  // tile's properties
  const int height = 2 * radiusXY;
  const int width = 2 * radiusXY;
  const int tileSize = width*height;
  const int lineSizeInBytes = 2 * radiusXY*sizeof(float);

  // image's properties
  const auto& imageSize = image->GetLargestPossibleRegion().GetSize();
  const int sliceSize = imageSize[0] * imageSize[1];
  const int lineSize = imageSize[1];

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);

  for (int i = 0; i < totalCount / batchLength; ++i) { //todo fix last total%batch_size indices
    auto time0 = clock();
    std::cout << i << "th batch" << std::endl;

    Blob<float>* blob = new Blob<float>(batchLength, channels, height, width); // has been moved out from the loop

    //int lastX = -1; // we dont use last z and assume thay 'z' stay same
    for (int iTile = 0; iTile < batchLength; ++iTile) {
      const auto& index = indices[i*batchLength + iTile];
      //if (index[0] == lastX) { // x same as before
      //concat new line(s) to last tile
      //} else {
      //make tile from the scratch
      const int zOffset = index[2] * sliceSize;
      const int xOffset = (index[0] - radiusXY + 1);
      const int yOffsetPart = (index[1] - radiusXY + 1)* lineSize;
      float* src = buffer + zOffset + yOffsetPart + xOffset;

      const int tileOffset = iTile*tileSize;
      float* dst = blob->mutable_cpu_data() + tileOffset;

      for (int iRow = 0; iRow < 2 * radiusXY; iRow++) { // try to compute offset by 1 vector command
        memcpy(dst, src, lineSizeInBytes);

        src += lineSize; // adjust yOffset
        dst += width; // adjust lineOffset
      }
    }

    //fill the vector
    vector<Blob<float>*> bottom;
    bottom.push_back(blob);
    float type = 0.0;

    auto time1 = clock();

    auto results = caffe_test_net.Forward(bottom, &type)[0]->cpu_data();
    delete blob;

    auto time2 = clock();
    //Here I can use the argmax layer, but for now I do a simple for :)
    for (int iTile = 0; iTile < batchLength; ++iTile) {
      auto& index = indices[i*batchLength + iTile];

      float max = 0;
      int max_i = 0;
      for (int j = 0; j < classCount; ++j) {
        float value = results[classCount*iTile + j];
        if (value > max) {
          max = value;
          max_i = j;
        }
        outImages[j]->SetPixel(index, round(value));
      }

      int val = 0;

      if (classCount == 4) {
        const int TP = 2, FN = 3; // there are labels from last classificatoin onto 2 classes

        if (max_i == TP || max_i == FN) { // TP,FN -> true, TN,FP ->false
          val = 1;
          tumCount++;
        }
      }
      else if (classCount == 3) { // there are background, singularity in tumors, tumors
        if (max_i != 0) {
          val = 1;
          tumCount++;
        }
      }
      else {// if classCount == 2
        if (max_i == 1) {
          val = 1;
          tumCount++;
        }
      }
      //set group's area
      for (int k = -groupX / 2; k < groupX - groupX / 2; ++k) {
        for (int l = -groupY / 2; l < groupY - groupY / 2; ++l) {
          agtk::Image3DSize offset = {k, l, 0};
          auto index2 = index + offset;
          outImage->SetPixel(index2, val); // can be improved if only group 1x1 used
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

  std::cout << "tumors - " << tumCount << "\n";

  typedef itk::ImageFileWriter<agtk::BinaryImage3D>  writerType;
  writerType::Pointer writer = writerType::New();
  writer->SetFileName(output_file);
  writer->SetInput(outImage);
  try {
    writer->Update();
  }
  catch (itk::ExceptionObject &excp) {
    std::cout << "Exception thrown while writing " << std::endl;
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < classCount; i++) {
    writerType::Pointer writerI = writerType::New();
    writerI->SetFileName(output_file + std::to_string(i) + ".nrrd");
    writerI->SetInput(outImages[i]);
    writerI->Update();
  }


  return EXIT_SUCCESS;
}
