#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>

// ITK
#include <itkImage.h>
#include <itkMetaImageIOFactory.h>
#include <itkNrrdImageIOFactory.h>

#include "caffe/caffe.hpp"

#include "agtkTypes.h"
#include "agtkIO.h"

#include "../caffefication.h"

//#define RUNTIME_CAFFEFICATION

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


#ifdef RUNTIME_CAFFEFICATION // just for test/debug
  auto m_Dll = LoadLibrary("D:/alex/CNN-build4/Release/caffefication.dll");

  // Check to see if the library was loaded successfully 
  if (!m_Dll) {
    printf("library failed to load!\n");
    return EXIT_FAILURE;
  }

  //declare a variable of type pointer to EntryPoint function, a name of 
  // which you will later use instead of EntryPoint
  auto loadNetStr = "loadNet";
  loadNetFuncPtr loadNetFunc = reinterpret_cast<loadNetFuncPtr>(GetProcAddress(m_Dll, loadNetStr));
  if (!loadNetFunc) {
    std::cout << "no function " << loadNetStr << std::endl;
    return EXIT_FAILURE;
  }

  auto classifyStr = "classify";
  auto m_ClassifyFunc = reinterpret_cast<classifyFuncPtr>(GetProcAddress(m_Dll, classifyStr));
  if (!m_ClassifyFunc) {
    std::cout << "no function " << classifyStr << std::endl;
    return EXIT_FAILURE;
  }
#define classify m_ClassifyFunc
#define loadNet loadNetFunc
#endif 
  std::shared_ptr<caffe::Net<float>> caffeNet;
  
  //Setting CPU or GPU
  caffefication::loadNet(modelFile, trainedFile, deviceId, caffeNet);
  std::cout << "load images" << std::endl;

  itk::MetaImageIOFactory::RegisterOneFactory();
  itk::NrrdImageIOFactory::RegisterOneFactory();
  std::cout << "nrrd factory registered" << std::endl;

  typedef std::list<itk::LightObject::Pointer> RegisteredObjectsContainerType;
  RegisteredObjectsContainerType registeredIOs = itk::ObjectFactoryBase::CreateAllInstance("itkImageIOBase");
  std::cout << "there are " << registeredIOs.size() << " IO objects available to the ImageFileReader." << std::endl;

  std::cout << __LINE__ << std::endl;  //dbg
  BinaryImage3D::Pointer imageMask;
  if (maskFile == "BOUNDING_BOX") {
    imageMask = nullptr;
  } else {
    imageMask = BinaryImage3D::New();
    if (!readImage<BinaryImage3D>(imageMask, maskFile)) {
      std::cout << "Exception thrown while reading the mask " << maskFile << std::endl;
      return false;
    }
  }

  Int16Image3D::Pointer image16 = Int16Image3D::New();
  if (!readImage<Int16Image3D>(image16, inputFile)) {
    std::cout << "Exception thrown while reading the image " << inputFile << std::endl;
    return false;
  }

  BinaryImage3D::Pointer outImage;

  int success = caffefication::classify(caffeNet.get(), preset, image16, imageMask, region, radiusXY, spacingXY, batchLength, groupX, groupY, classCount, isRgb, outImage);

  if (!success) {
    return EXIT_FAILURE;
  }

  std::cout << "save" << std::endl;
  if (!writeImage(outImage, outputFile)) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
