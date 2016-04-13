#include <cuda_runtime.h>

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

#include "caffe/caffe.hpp"

#include "agtkTypes.h"

#include "D:\alex\CNN\caffefication.h" //TODO

bool writeImage(const std::string& outputFile, const agtk::BinaryImage3D::Pointer& outImage)
{
  typedef itk::ImageFileWriter<agtk::BinaryImage3D>  writerType;
  writerType::Pointer writer = writerType::New();
  writer->SetFileName(outputFile);
  writer->SetInput(outImage);
  try {
    writer->Update();
  } catch (itk::ExceptionObject &excp) {
    std::cout << "Exception thrown while writing " << std::endl;
    std::cout << excp << std::endl;
    return false;
  }
  return true;
}

int main(int argc, char** argv)
{
  using namespace caffe;
  using namespace agtk;

  string modelFile = argv[1];
  string trainedFile = argv[2];

  string startXStr = argv[3];
  string startYStr = argv[4];
  string startZStr = argv[5];

  string sizeXStr = argv[6];
  string sizeYStr = argv[7];
  string sizeZStr = argv[8];

  string radiusXYStr = argv[9];
  string preset = argv[10];
  string spacingXYStr = argv[11];

  string batchLengthStr = argv[12];

  string groupXStr = argv[13]; // interpret an area XxY as 1 unit
  string groupYStr = argv[14];

  string classCountStr = argv[15];

  string inputFile = argv[16];
  string maskFile = argv[17];
  string outputFile = argv[18];

  string deviceIdStr = argv[19];

  Image3DIndex start;
  start[0] = atoi(startXStr.c_str());
  start[1] = atoi(startYStr.c_str());
  start[2] = atoi(startZStr.c_str());

  Image3DSize size;
  size[0] = atoi(sizeXStr.c_str());
  size[1] = atoi(sizeYStr.c_str());
  size[2] = atoi(sizeZStr.c_str());

  Image3DRegion region;
  region.SetIndex(start);
  region.SetSize(size);

  int radiusXY = atoi(radiusXYStr.c_str());
  float spacingXY = atof(spacingXYStr.c_str());
  int batchLength = atoi(batchLengthStr.c_str());
  int groupX = atoi(groupXStr.c_str());
  int groupY = atoi(groupYStr.c_str());
  int classCount = atoi(classCountStr.c_str());
  int deviceId = atoi(deviceIdStr.c_str());
  bool isRgb = false; //TODO

  std::cout << "modelFile = " << modelFile << std::endl <<
    "trainedFile =" << trainedFile << std::endl <<
    "region = " << region << std::endl <<
    "radiusXY=" << radiusXY << std::endl <<
    "preset=" << preset << std::endl <<
    "spacingXY=" << spacingXY << std::endl <<
    "batchSize=" << batchLength << std::endl <<
    "groupX=" << groupX << std::endl <<
    "groupY=" << groupY << std::endl <<
    "classCount=" << classCount << std::endl <<
    "inputFile = " << inputFile << std::endl <<
    "maskFile =" << maskFile << std::endl <<
    "outputFile =" << outputFile << std::endl <<
    "deviceID =" << deviceId << std::endl;

  //Setting CPU or GPU
  std::shared_ptr<caffe::Net<float>> caffeNet;
  loadNet(modelFile, trainedFile, deviceId, caffeNet);

  std::cout << "load images" << std::endl;

  itk::MetaImageIOFactory::RegisterOneFactory();
  itk::NrrdImageIOFactory::RegisterOneFactory();
  std::cout << "nrrd factory registered" << std::endl;

  typedef std::list<itk::LightObject::Pointer> RegisteredObjectsContainerType;
  RegisteredObjectsContainerType registeredIOs = itk::ObjectFactoryBase::CreateAllInstance("itkImageIOBase");
  std::cout << "there are " << registeredIOs.size() << " IO objects available to the ImageFileReader." << std::endl;

  typedef itk::ImageFileReader<Int16Image3D> ReaderType;
  typedef itk::ImageFileReader<BinaryImage3D> BinaryReaderType;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFile);
  try {
    reader->Update();
  } catch (itk::ExceptionObject &excp) {
    std::cout << "Exception thrown while reading the image " << std::endl;
    std::cout << excp << std::endl;
    return false;
  }
  std::cout << "." << std::endl;

  BinaryImage3D::Pointer imageMask;
  if (maskFile == "BOUNDING_BOX") {
    imageMask = nullptr;
  } else {
    BinaryReaderType::Pointer readerMask = BinaryReaderType::New();
    readerMask->SetFileName(maskFile);
    try {
      readerMask->Update();
    } catch (itk::ExceptionObject &excp) {
      std::cout << "Exception thrown while reading the mask " << std::endl;
      std::cout << excp << std::endl;
      return false;
    }
    imageMask = readerMask->GetOutput();
  }
  std::cout << "." << std::endl;

  Int16Image3D::Pointer image16 = reader->GetOutput();

  BinaryImage3D::Pointer outImage;
  int success = classify(caffeNet.get(), preset, image16, imageMask, region, radiusXY, spacingXY, batchLength, groupX, groupY, classCount, isRgb, outImage);
  if (!success) {
    return EXIT_FAILURE;
  }

  std::cout << "save" << std::endl;
  if (!writeImage(outputFile, outImage)) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
