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
// performs resampling of input image to specified spacing
// 0 in outSpacing[i] means spacing by the dimension stays same 
template <typename TImage>
typename TImage::Pointer resample(const TImage* image, typename TImage::SpacingType outSpacing)
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

// non-const overload-wrapper
template <typename TImage>
typename TImage::Pointer resample(TImage* image, typename TImage::SpacingType outSpacing)
{
  return resample(const_cast<const TImage*>(image), outSpacing);
}

// Use this function if need good binary image as result
template <unsigned int VDim, typename TPixel>
typename itk::Image<TPixel, VDim>::Pointer resampleBinary(itk::Image<TPixel, VDim>* image, typename itk::ImageBase<VDim>::SpacingType outSpacing)
{
  typedef itk::Image<TPixel, VDim> ImageType;

  typename ImageType::SizeType inSize = image->GetLargestPossibleRegion().GetSize();
  typename ImageType::SpacingType inSpacing = image->GetSpacing();
  typename ImageType::SizeType outSize;

  for (int n = 0; n < VDim; ++n) {
    if (outSpacing[n] > 0) {
      outSize[n] = inSize[n] * (inSpacing[n] / outSpacing[n]) - 1;
    } else {
      outSpacing[n] = inSpacing[n];
      outSize[n] = inSize[n];
    }
  }

  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

  typedef itk::NearestNeighborInterpolateImageFunction<ImageType> Interpolator;
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

// Use this function if image with exactly same geometry as result
template <typename TImage>
typename TImage::Pointer resample(const TImage* image, const TImage* referenceImage)
{
  typedef TImage ImageType;

  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

  auto interpolator = itk::NearestNeighborInterpolateImageFunction<ImageType>::New();

  auto resampleFilter = ResampleImageFilterType::New();
  resampleFilter->SetInput(image);
  resampleFilter->SetReferenceImage(referenceImage);
  resampleFilter->SetUseReferenceImage(true);
  resampleFilter->SetInterpolator(interpolator);
  resampleFilter->SetDefaultPixelValue(0);

  resampleFilter->UpdateLargestPossibleRegion();
  return resampleFilter->GetOutput();
}

//----------------------------------------------------------------------------
template <unsigned int VDim, typename TPixel>
typename itk::Image <TPixel, VDim>::Pointer resize(const itk::Image <TPixel, VDim>* image, typename itk::ImageBase<VDim>::SizeType outSize)
{
  typedef itk::Image <TPixel, VDim> ImageType;

  auto inSize = image->GetLargestPossibleRegion().GetSize();
  typename ImageType::SpacingType outSpacing;
  for (size_t i = 0; i < VDim; i++) {
    outSpacing[i] = image->GetSpacing()[i] * inSize[i] / outSize[i];
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
template <unsigned int VDim, typename TPixel>
typename itk::Image <TPixel, VDim>::Pointer resizeBinary(const itk::Image <TPixel, VDim>* image, typename itk::ImageBase<VDim>::SizeType outSize)
{
  typedef itk::Image <TPixel, VDim> ImageType;

  auto inSize = image->GetLargestPossibleRegion().GetSize();
  typename ImageType::SpacingType outSpacing;
  for (size_t i = 0; i < VDim; i++) {
    outSpacing[i] = image->GetSpacing()[i] * inSize[i] / outSize[i];
  }

  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;

  typedef itk::NearestNeighborInterpolateImageFunction<ImageType> Interpolator;
  auto interpolator = Interpolator::New();

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

}

#endif // __agtkResampling_h
