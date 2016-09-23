#ifndef __agtkIO_h
#define __agtkIO_h

#include <string>
#include <iostream>

#include <itkObject.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMeshFileReader.h>
#include <itkMeshFileWriter.h>

#include "agtkTypes.h"
#include "agtkCoreExport.h"

namespace agtk
{
//! Reads a templated image from a file via ITK ImageFileReader
template <typename TImage>
bool readImage(typename TImage::Pointer image, const std::string& fileName)
{
  typedef itk::ImageFileReader<TImage> Reader;

  typename Reader::Pointer reader = Reader::New();

  reader->SetFileName(fileName);

  try {
    reader->Update();
  }
  catch (itk::ExceptionObject& err) {
    std::cerr << "Unable to read image from file '" << fileName << "'" << std::endl;
    std::cerr << "Error: " << err << std::endl;
    return false;
  }

  image->Graft(reader->GetOutput());
  return true;
}

//! Writes a templated image to a file via ITK ImageFileWriter
template <typename TImage>
bool writeImage(const TImage* image, const std::string& fileName)
{
  typedef itk::ImageFileWriter<TImage> Writer;

  typename Writer::Pointer writer = Writer::New();

  writer->SetInput(image);
  writer->SetFileName(fileName);
  writer->SetUseCompression(true);

  try {
    writer->Update();
  }
  catch (itk::ExceptionObject& err) {
    std::cerr << "Unable to write image to file '" << fileName << "'" << std::endl;
    std::cerr << "Error: " << err << std::endl;
    return false;
  }

  return true;
}

//! Reads a mesh from a file
template <typename TMesh>
bool readMesh(typename TMesh::Pointer mesh, const std::string& fileName)
{
  typedef itk::MeshFileReader<TMesh> MeshFileReader;
  typename MeshFileReader::Pointer reader = MeshFileReader::New();
  reader->SetFileName(fileName);

  try {
    reader->Update();
  }
  catch (itk::ExceptionObject& err) {
    std::cerr << "Unable to read mesh from file '" << fileName << "'" << std::endl;
    std::cerr << "Error: " << err << std::endl;
    return false;
  }

  mesh->Graft(reader->GetOutput());
  return true;
}

//! Writes a mesh to a file
template <typename TMesh>
bool writeMesh(const TMesh* mesh, const std::string& fileName)
{
  typedef itk::MeshFileWriter<TMesh> MeshFileWriter;
  typename MeshFileWriter::Pointer writer = MeshFileWriter::New();

  writer->SetFileName(fileName);
  writer->SetInput(mesh);
  writer->SetUseCompression(true);
  writer->SetFileTypeAsBINARY();

  try {
    writer->Update();
  }
  catch (itk::ExceptionObject& err) {
    std::cerr << "Unable to write mesh to file '" << fileName << "'" << std::endl;
    std::cerr << "Error: " << err << std::endl;
    return false;
  }

  return true;
}

//! Reads an image 3D as float from a file
AGTKCore_EXPORT bool readImage(FloatImage3D::Pointer image, const std::string& fileName);

//! Reads an image 3D as uint8 from a file
AGTKCore_EXPORT bool readImage(UInt8Image3D::Pointer image, const std::string& fileName);

//! Writes an uint8 image 3D to a file
AGTKCore_EXPORT bool writeImage(const UInt8Image3D* image, const std::string& fileName);

//! Writes an float image 3D to a file
AGTKCore_EXPORT bool writeImage(const FloatImage3D* image, const std::string& fileName);


//! Class writes some data to a directory
class AGTKCore_EXPORT DataWriter : public itk::Object
{
public:
  typedef DataWriter Self;
  typedef itk::Object Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  itkNewMacro(Self);
  itkTypeMacro(DataWriter, itk::Object);

  itkSetMacro(Enabled, bool);
  itkGetMacro(Enabled, bool);
  itkBooleanMacro(Enabled);

  bool SetOutputDirectory(const std::string& dirName);

  itkGetStringMacro(OutputDirectory);

  itkGetStringMacro(FileNamePrefix);
  itkSetStringMacro(FileNamePrefix);

  itkSetStringMacro(FileNamePrefixSeparator);
  itkGetStringMacro(FileNamePrefixSeparator);

  itkSetStringMacro(ImageFileExtension);
  itkGetStringMacro(ImageFileExtension);

  //--------------------------------------------------------------------------
  template <typename TImage>
  bool WriteImage(const TImage* image, const std::string& fileName)
  {
    if (!m_Enabled) {
      return false;
    }

    return writeImage<TImage>(image, GetFullFileName(fileName) + m_ImageFileExtension);
  }

  //--------------------------------------------------------------------------
  bool WriteImage(const FloatImage3D* image, const std::string& fileName)
  {
    return WriteImage<FloatImage3D>(image, fileName);
  }

  //--------------------------------------------------------------------------
  bool WriteImage(const BinaryImage3D* image, const std::string& fileName)
  {
    return WriteImage<BinaryImage3D>(image, fileName);
  }

protected:
  DataWriter();
  virtual ~DataWriter() {}

  bool m_Enabled;
  std::string m_OutputDirectory;

  std::string m_FileNamePrefix;
  std::string m_ImageFileExtension;
  std::string m_FileNamePrefixSeparator;

private:
  DataWriter(const Self&);
  void operator=(const Self&);

  bool CreateDirectoryIfNotExists();
  std::string GetFullFileName(const std::string& fileName);
};
}

#endif // __agtkIO_h
