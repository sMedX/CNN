#include <boost/filesystem.hpp>

#include "agtkIO.h"

namespace fs = boost::filesystem;

namespace agtk
{
//----------------------------------------------------------------------------
bool readImage(FloatImage3D::Pointer image, const std::string& fileName)
{
  return readImage<FloatImage3D>(image, fileName);
}

//----------------------------------------------------------------------------
bool readImage(UInt8Image3D::Pointer image, const std::string& fileName)
{
  return readImage<UInt8Image3D>(image, fileName);
}

//----------------------------------------------------------------------------
bool writeImage(const UInt8Image3D* image, const std::string& fileName)
{
  return writeImage<UInt8Image3D>(image, fileName);
}

//----------------------------------------------------------------------------
bool writeImage(const FloatImage3D* image, const std::string& fileName)
{
  return writeImage<FloatImage3D>(image, fileName);
}

//----------------------------------------------------------------------------
DataWriter::DataWriter()
{
  m_Enabled = true;
  m_OutputDirectory = std::string("");

  m_FileNamePrefix = "";
  m_FileNamePrefixSeparator = "_";
  m_ImageFileExtension = ".nrrd";
}

//----------------------------------------------------------------------------
bool DataWriter::SetOutputDirectory(const std::string& dirName)
{
  m_OutputDirectory = dirName;
  return CreateDirectoryIfNotExists();
}

//----------------------------------------------------------------------------
bool DataWriter::CreateDirectoryIfNotExists()
{
  fs::path outputDir(m_OutputDirectory);

  if (fs::is_directory(outputDir)) {
    return true;
  }

  return fs::create_directories(outputDir);
}

//----------------------------------------------------------------------------
std::string DataWriter::GetFullFileName(const std::string& fileName)
{
  std::string prefix = (m_FileNamePrefix.size() > 0) ? (m_FileNamePrefix + m_FileNamePrefixSeparator) : "";

  fs::path outputDir(m_OutputDirectory);
  fs::path fName(prefix + fileName);
  fs::path fullFileName = outputDir / fName;

  return fullFileName.string();
}
}
