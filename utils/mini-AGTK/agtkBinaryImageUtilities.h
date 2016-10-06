/*******************************************************************************
*
* This file is a part of AGTK
*
*
* Copyright 2015, SMedX LLC
*
*******************************************************************************/
#ifndef __agtkBinaryImageUtilities_h
#define __agtkBinaryImageUtilities_h

#include <itkBinaryThresholdImageFilter.h>

#include "agtkTypes.h"

#define AGTK_EXPORT  //#include "agtkExport.h"

namespace agtk
{
template <typename TInputImage>
typename BinaryImage3D::Pointer binarizeImage(const TInputImage* image,
                                              typename TInputImage::PixelType lowerThreshold, typename TInputImage::PixelType upperThreshold)
{
  typedef itk::BinaryThresholdImageFilter<TInputImage, BinaryImage3D> ThresholdImageFilter;
  typename ThresholdImageFilter::Pointer thresholder = ThresholdImageFilter::New();

  thresholder->SetInput(image);

  thresholder->SetInsideValue(INSIDE_BINARY_VALUE);
  thresholder->SetOutsideValue(OUTSIDE_BINARY_VALUE);
  thresholder->SetLowerThreshold(lowerThreshold);
  thresholder->SetUpperThreshold(upperThreshold);

  thresholder->Update();

  return thresholder->GetOutput();
}

//! It computes binary image from float image by threshold
AGTK_EXPORT BinaryImage3D::Pointer binarizeImage(const FloatImage3D* image, float lowerThreshold, float upperThreshold);

//! It computes binary image from float image by threshold
AGTK_EXPORT BinaryImage3D::Pointer binarizeImage(const FloatImage3D* image, float upperThreshold);

//! Merges two images
AGTK_EXPORT BinaryImage3D::Pointer mergeTwoBinaryImages(const BinaryImage3D* image1, const BinaryImage3D* image2);

//! Returns mask bounding box region
AGTK_EXPORT Image3DRegion getBinaryMaskBoundingBoxRegion(const BinaryImage3D* image, bool* isValidRegion = nullptr);

//! Returns cropped binary image by bounding box mask region
AGTK_EXPORT BinaryImage3D::Pointer cropBinaryImageByMaskBoundingBox(const BinaryImage3D* image,
                                                                        Image3DRegion* region = nullptr, unsigned int padding = 0);

//! Returns largest connected component from binary image
AGTK_EXPORT BinaryImage3D::Pointer getLargestObjectFromBinaryImage(const BinaryImage3D* image);

//! Fills holes in a binary objects
AGTK_EXPORT BinaryImage3D::Pointer fillHolesOnBinaryImage(const BinaryImage3D* image, bool fullyConnected = true);

const Image3DSize __defaultRadius = { 1, 1, 1 };

//! Fills holes and cavities on a binary image
AGTK_EXPORT BinaryImage3D::Pointer iterativeFillHolesAndCavitiesOnBinaryImage(const BinaryImage3D* image,
                                                                                  unsigned int maximumIterations = 10, const Image3DSize& radius = __defaultRadius, unsigned int majorityThreshold = 1);
}

#endif // __agtkBinaryImageUtilities_h
