/*******************************************************************************
*
* This file is a part of AGTK
*
*
* Copyright 2015, SMedX LLC
*
*******************************************************************************/

#include <itkNumericTraits.h>
#include <itkMaximumImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkLabelShapeKeepNObjectsImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkBinaryFillholeImageFilter.h>
#include <itkVotingBinaryIterativeHoleFillingImageFilter.h>

#include "agtkImageFilteringShortcuts.h"
#include "agtkBinaryImageUtilities.h"

namespace agtk
{
//--------------------------------------------------------------------------
BinaryImage3D::Pointer binarizeImage(const FloatImage3D* image, float lowerThreshold, float upperThreshold)
{
  return binarizeImage<FloatImage3D>(image, lowerThreshold, upperThreshold);
}

//--------------------------------------------------------------------------
BinaryImage3D::Pointer binarizeImage(const FloatImage3D* image, float upperThreshold)
{
  float lowerThreshold = FloatLimits::lowest();

  return binarizeImage(image, lowerThreshold, upperThreshold);
}

//--------------------------------------------------------------------------
BinaryImage3D::Pointer mergeTwoBinaryImages(const BinaryImage3D* image1, const BinaryImage3D* image2)
{
  typedef itk::MaximumImageFilter<BinaryImage3D> MaximumImageFilter;
  MaximumImageFilter::Pointer maxImage = MaximumImageFilter::New();

  maxImage->SetInput1(image1);
  maxImage->SetInput2(image2);
  maxImage->Update();

  return maxImage->GetOutput();
}

//----------------------------------------------------------------------------
agtk::Image3DRegion getBinaryMaskBoundingBoxRegion(const BinaryImage3D* image, bool* isValidRegion)
{
  Image3DRegion region = thresholdBoundingBoxRegion<BinaryImage3D>(image, 1, BinaryLimits::max());

  if (isValidRegion != nullptr) {
    *isValidRegion = true;
    Image3DSize sz = region.GetSize();

    for (int i = 0; i < IMAGE_DIM_3; ++i) {
      if (sz[i] == 0) {
        *isValidRegion = false;
        break;
      }
    }
  }

  return region;
}

//--------------------------------------------------------------------------
BinaryImage3D::Pointer cropBinaryImageByMaskBoundingBox(const BinaryImage3D* image, Image3DRegion* region /*=nullptr*/, unsigned int padding /*=0*/)
{
  bool isValidRegion = false;
  Image3DRegion r = getBinaryMaskBoundingBoxRegion(image, &isValidRegion);

  BinaryImage3D::Pointer croppedImage = const_cast<BinaryImage3D*>(image);

  if (isValidRegion) {
    if (padding > 0) {
      Image3DOffset offset;
      offset.Fill(padding);

      Image3DIndex lIndex = r.GetIndex() - offset;
      Image3DIndex uIndex = r.GetUpperIndex() + offset;

      r.SetIndex(lIndex);
      r.SetUpperIndex(uIndex);
      r.Crop(image->GetLargestPossibleRegion());
    }

    if (r.GetSize() != image->GetLargestPossibleRegion().GetSize()) {
      croppedImage = cropImage<BinaryImage3D>(image, r);
    }
  }
  else {
    r = image->GetLargestPossibleRegion();
  }

  if (region != nullptr) {
    region->SetIndex(r.GetIndex());
    region->SetSize(r.GetSize());
  }

  return croppedImage;
}

//--------------------------------------------------------------------------
BinaryImage3D::Pointer getLargestObjectFromBinaryImage(const BinaryImage3D* image)
{
  typedef itk::ConnectedComponentImageFilter<UInt8Image3D, UInt16Image3D> ConnectedComponentImageFilter;
  typedef itk::LabelShapeKeepNObjectsImageFilter<UInt16Image3D> LabelShapeKeepNObjectsImageFilter;
  typedef itk::RescaleIntensityImageFilter<UInt16Image3D, BinaryImage3D> RescaleIntensityImageFilter;

  ConnectedComponentImageFilter::Pointer connected = ConnectedComponentImageFilter::New();

  connected->SetInput(image);
  connected->Update();

  LabelShapeKeepNObjectsImageFilter::Pointer labelShape = LabelShapeKeepNObjectsImageFilter::New();

  labelShape->SetInput(connected->GetOutput());
  labelShape->SetBackgroundValue(OUTSIDE_BINARY_VALUE);
  labelShape->SetNumberOfObjects(1);
  labelShape->SetAttribute(LabelShapeKeepNObjectsImageFilter::LabelObjectType::NUMBER_OF_PIXELS);

  RescaleIntensityImageFilter::Pointer rescaler = RescaleIntensityImageFilter::New();

  rescaler->SetOutputMinimum(OUTSIDE_BINARY_VALUE);
  rescaler->SetOutputMaximum(INSIDE_BINARY_VALUE);
  rescaler->SetInput(labelShape->GetOutput());
  rescaler->Update();

  return rescaler->GetOutput();
}

//----------------------------------------------------------------------------
BinaryImage3D::Pointer fillHolesOnBinaryImage(const BinaryImage3D* image, bool fullyConnected)
{
  typedef itk::BinaryFillholeImageFilter<BinaryImage3D> Fillhole;
  Fillhole::Pointer fillhole = Fillhole::New();

  fillhole->SetInput(image);
  fillhole->SetFullyConnected(fullyConnected);
  fillhole->Update();

  return fillhole->GetOutput();
}

//----------------------------------------------------------------------------
BinaryImage3D::Pointer iterativeFillHolesAndCavitiesOnBinaryImage(const BinaryImage3D* image,
                                                                  unsigned int maximumIterations, const Image3DSize& radius, unsigned int majorityThreshold)
{
  typedef itk::VotingBinaryIterativeHoleFillingImageFilter<BinaryImage3D> Fillhole;
  Fillhole::Pointer fillhole = Fillhole::New();

  fillhole->SetInput(image);
  fillhole->SetMaximumNumberOfIterations(maximumIterations);
  fillhole->SetMajorityThreshold(majorityThreshold);
  fillhole->SetRadius(radius);
  fillhole->SetBackgroundValue(OUTSIDE_BINARY_VALUE);
  fillhole->SetForegroundValue(INSIDE_BINARY_VALUE);
  fillhole->Update();

  return fillhole->GetOutput();
}
}
