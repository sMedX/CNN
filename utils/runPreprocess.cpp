#include <iostream>

#include <itkImage.h>
#include <itkMetaImageIOFactory.h>
#include <itkNrrdImageIOFactory.h>

#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkCommandLineArgumentParser.h"

#include "preprocess.h"

int main(int argc, char* argv[])
{
  using namespace agtk;
  using namespace caffefication;

  itk::MetaImageIOFactory::RegisterOneFactory();
  itk::NrrdImageIOFactory::RegisterOneFactory();
  std::cout << "nrrd factory registered" << std::endl;

  typedef std::list<itk::LightObject::Pointer> RegisteredObjectsContainerType;
  RegisteredObjectsContainerType registeredIOs = itk::ObjectFactoryBase::CreateAllInstance("itkImageIOBase");
  std::cout << "there are " << registeredIOs.size() << " IO objects available to the ImageFileReader." << std::endl;

  /** Create a command line argument parser. */
  CommandLineArgumentParser::Pointer parser = CommandLineArgumentParser::New();
  parser->SetCommandLineArguments(argc, argv);

  std::string inputName;
  parser->GetValue("--input", inputName);

  std::string preset;
  parser->GetValue("--preset", preset);

  bool isBinary = false;
  parser->GetValue("--binary", isBinary);

  Image3DSpacing spacing;
  spacing.Fill(0);
  parser->GetITKValue<Image3DSpacing>("--spacing", spacing);

  std::string outputName;
  parser->GetValue("--output", outputName);

  std::cout << "input   " << inputName << std::endl;
  std::cout << "preset " << preset << std::endl;
  std::cout << "output   " << outputName << std::endl;
  std::cout << "binary   " << isBinary << std::endl;
  std::cout << "spacing   " << spacing << std::endl;

  // it's necassary
  int radius = 0;
  bool isRgb = false;

  BinaryImage3D::Pointer output;

  // read image
  std::cout << "load image" << std::endl;
  if (isBinary) {
    BinaryImage3D::Pointer image = BinaryImage3D::New();
    if (!readImage(image, inputName)) {
      std::cout << "can't read " << inputName;
      return EXIT_FAILURE;
    }
    output = preprocessBinary(radius, spacing, isRgb, image);

  } else {
    Int16Image3D::Pointer image16 = Int16Image3D::New();
    if (!readImage<Int16Image3D>(image16, inputName)) {
      std::cout << "can't read " << outputName;
      return EXIT_FAILURE;
    }

    // organ-based transformation to UINT8 from int16
    auto image = smartCastImage(preset, image16, nullptr);
    output = preprocess(radius, spacing, isRgb, image);
  }

  writeImage(output.GetPointer(), outputName);

  return EXIT_SUCCESS;
};
