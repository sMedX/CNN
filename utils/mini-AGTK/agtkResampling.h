#ifndef __agtkResampling_h
#define __agtkResampling_h

#include <itkIdentityTransform.h>
#include <itkResampleImageFilter.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <itkConstantBoundaryCondition.h>

#include "agtkTypes.h"

namespace agtk
{
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
    }
    else {
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

//----------------------------------------------------------------------------
template <typename TImage>
typename TImage::Pointer resampling(TImage* image, typename TImage::SpacingType outSpacing)
{
  return resampling(const_cast<const TImage*>(image), outSpacing);
}

// Use this function if need binary image as result
template <unsigned int VDim>
typename itk::Image<UInt8Pixel, VDim>::Pointer resamplingBinary(itk::Image<UInt8Pixel, VDim>* image,
  typename itk::ImageBase<VDim>::SpacingType outSpacing)
{
  typedef itk::Image<UInt8Pixel, VDim> ImageType;

  typename ImageType::SizeType inSize = image->GetLargestPossibleRegion().GetSize();
  typename ImageType::SpacingType inSpacing = image->GetSpacing();
  typename ImageType::SizeType outSize;

  for (int n = 0; n < VDim; ++n) {
    if (outSpacing[n] > 0) {
      outSize[n] = inSize[n] * (inSpacing[n] / outSpacing[n]) - 1;
    }
    else {
      outSpacing[n] = inSpacing[n];
      outSize[n] = inSize[n];
    }
  }

  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

  typedef itk::NearestNeighborInterpolateImageFunction<BinaryImage3D> Interpolator;
  auto interpolator = Interpolator::New();

  typename ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
  resample->SetDefaultPixelValue(0);
  resample->SetInterpolator(interpolator);
  resample->SetOutputSpacing(outSpacing);
  resample->SetSize(outSize);
  resample->SetOutputOrigin(image->GetOrigin());
  resample->SetInput(image);
  resample->Update();

  BinaryImage3D::Pointer output = resample->GetOutput();

  return output;
}

template <typename TImage>
typename TImage::Pointer resamplingLike(const TImage* image, const TImage* referenceImage)
{
  typedef TImage ImageType;

  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

  auto nn_interpolator = itk::NearestNeighborInterpolateImageFunction<ImageType>::New();

  auto resampleFilter = ResampleImageFilterType::New();
  resampleFilter->SetInput(image);
  resampleFilter->SetReferenceImage(referenceImage);
  resampleFilter->SetUseReferenceImage(true);
  resampleFilter->SetInterpolator(nn_interpolator);
  resampleFilter->SetDefaultPixelValue(0);

  resampleFilter->UpdateLargestPossibleRegion();
  return resampleFilter->GetOutput();
}
}

#endif // __agtkResampling_h
