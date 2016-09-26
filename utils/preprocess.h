#include <itkConstantPadImageFilter.h>

#include <itkResampleImageFilter.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <itkConstantBoundaryCondition.h>

#include "agtkTypes.h"
#include "agtkResampling.h"

namespace caffefication {
using namespace agtk;

//----------------------------------------------------------------------------
// 0 in outSpacing mean that this axis will not resampled
template <typename TImage>
typename TImage::Pointer resampling(const TImage* image, typename TImage::SpacingType outSpacing)
{
  typedef TImage ImageType;

  typename ImageType::SizeType inSize = image->GetLargestPossibleRegion().GetSize();
  typename ImageType::SpacingType inSpacing = image->GetSpacing();
  typename ImageType::SizeType outSize;

  for (int n = 0; n < ImageType::ImageDimension; ++n) {
    if (outSpacing[n] > 0) {
      outSize[n] = inSize[n] * (inSpacing[n] / outSpacing[n]) - 1;
    } else {
      outSpacing[n] = inSpacing[n];
      outSize[n] = inSize[n];
    }
  }

  const unsigned int WindowRadius = 2;
  typedef itk::Function::HammingWindowFunction<WindowRadius> WindowFunctionType;
  typedef itk::ConstantBoundaryCondition<ImageType> BoundaryConditionType;
  typedef itk::WindowedSincInterpolateImageFunction<ImageType, WindowRadius, WindowFunctionType, BoundaryConditionType, double> InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

  typename ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
  resample->SetInterpolator(interpolator);
  resample->SetDefaultPixelValue(0);
  resample->SetOutputSpacing(outSpacing);
  resample->SetSize(outSize);
  resample->SetOutputOrigin(image->GetOrigin());
  resample->SetInput(image);
  resample->Update();

  typename ImageType::Pointer output = resample->GetOutput();

  return output;
}

inline BinaryImage3D::Pointer padImage(const BinaryImage3D* image, const itk::ImageBase<3>::SizeType& outputRegion)
{
  typedef BinaryImage3D ImageType;
  typedef itk::ConstantPadImageFilter <ImageType, ImageType> ConstantPadImageFilterType;

  const ImageType::PixelType constantPixel = 0;

  auto padFilter = ConstantPadImageFilterType::New();
  padFilter->SetInput(image);
  padFilter->SetPadBound(outputRegion); // Calls SetPadLowerBound(region) and SetPadUpperBound(region)
  padFilter->SetConstant(constantPixel);
  padFilter->Update();

  return padFilter->GetOutput();
}

// Performs preprocessing with casting to uint8
inline UInt8Image3D::Pointer smartCastImage(const std::string& preset, Int16Image3D* image16, BinaryImage3D* mask)
{
  std::cout << "shift, sqeeze" << std::endl;
  std::cout << "image16 " << image16 << std::endl;

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
    ////HARDCODE  normolizing image to mean 0 and std 1
    //double mean = 0;
    //int n = 0;
    //itk::ImageRegionConstIterator<BinaryImage3D> itMask(mask, mask->GetLargestPossibleRegion());
    //for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
    //  auto index = itMask.GetIndex();
    //  if (itMask.Get()) {
    //    mean += image16->GetPixel(itMask.GetIndex());
    //    ++n;
    //  }
    //}

    //if (n == 0) {
    //  auto error_empty_mask = "error: empty mask";
    //  std::cout << error_empty_mask;
    //  throw std::exception(error_empty_mask);
    //}

    //mean /= n;

    //std::cout << "mean is " << mean << std::endl;

    //double var = 0;
    //for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
    //  if (itMask.Get()) {
    //    var += pow(mean - image16->GetPixel(itMask.GetIndex()), 2);
    //  }
    //}
    //var /= (n - 1);

    //for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
    //  if (itMask.Get()) {
    //    auto index = itMask.GetIndex();
    //    image16->SetPixel(index, (image16->GetPixel(index) - mean) / var*1000 + 127); //
    //  }
    //}

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
  return cast->GetOutput();
}

// Performs preprocessing befory cutting by tiles
inline UInt8Image3D::Pointer preprocess(unsigned int radius, float spacingXY, bool isRgb, UInt8Image3D::Pointer input)
{
  UInt8Image3D::Pointer resampled = nullptr;
  //resample image by axial slices
  if (spacingXY != 0 && !(input->GetSpacing()[0] == spacingXY && input->GetSpacing()[1] == spacingXY))
  {
    std::cout << "resample. " << input->GetSpacing()[0] << " -> " << spacingXY << std::endl;
    Image3DSpacing spacing;
    spacing[0] = spacingXY;
    spacing[1] = spacingXY;
    spacing[2] = input->GetSpacing()[2];

    resampled = resampling(input.GetPointer(), spacing);
  }

  const Image3DSize size3D = { radius, radius, isRgb ? 1u : 0u };
  std::cout << "padding by radius " << size3D << std::endl;
  return padImage(resampled == nullptr ? input : resampled, size3D); // todo maybe use one call of resample filter for resampling and padding
}

// Performs preprocessing befory cutting by tiles
inline UInt8Image3D::Pointer preprocessBinary(unsigned int radius, float spacingXY, bool isRgb, UInt8Image3D::Pointer input)
{
  UInt8Image3D::Pointer resampled = nullptr;
  //resample image by axial slices
  if (spacingXY != 0 && !(input->GetSpacing()[0] == spacingXY && input->GetSpacing()[1] == spacingXY))
  {
    std::cout << "resample. " << input->GetSpacing()[0] << " -> " << spacingXY << std::endl;
    Image3DSpacing spacing;
    spacing[0] = spacingXY;
    spacing[1] = spacingXY;
    spacing[2] = input->GetSpacing()[2];

    resampled = resamplingBinary(input.GetPointer(), spacing);
  }

  const Image3DSize size3D = { radius, radius, isRgb ? 1u : 0u };
  std::cout << "padding by radius " << size3D << std::endl;
  return padImage(resampled == nullptr ? input : resampled, size3D); // todo maybe use one call of resample filter for resampling and padding
}
}
