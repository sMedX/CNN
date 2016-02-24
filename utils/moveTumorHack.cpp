#include <itkImage.h>

#include "agtkTypes.h"
#include "agtkIO.h"


int main(int argc, char* argv[])
{
  using namespace agtk;

  auto imageFile = "D:\\alex\\images\\022\\patient.nrrd";

  // read images
  std::cout << "load image" << std::endl;
  Int16Image3D::Pointer image16 = Int16Image3D::New();
  if (!readImage<Int16Image3D>(image16, imageFile)) {
    std::cout << "can't read " << imageFile;
    return EXIT_FAILURE;
  }

  agtk::Image3DSize size = {21, 19, 16};
  agtk::Image3DIndex indexFrom = {192, 262, 294}, indexTo = {133, 266, 266};
  agtk::Image3DRegion regionFrom = {indexFrom, size}, regionTo = {indexTo, size};

  itk::ImageRegionIterator<Int16Image3D> itFrom(image16, regionFrom);
  itk::ImageRegionIterator<Int16Image3D> itTo(image16, regionTo);

  for (itFrom.GoToBegin(), itTo.GoToBegin(); !itFrom.IsAtEnd(); ++itFrom, ++itTo) {
    itTo.Set(itFrom.Get());
  }

  writeImage(image16.GetPointer(), "D:\\alex\\images\\022\\patient_modified.nrrd");

  return EXIT_SUCCESS;
};
