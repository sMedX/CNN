/*******************************************************************************
*
* This file is a part of AGTK
*
*
* Copyright 2015, SMedX LLC
*
*******************************************************************************/
#ifndef __agtkImageFilteringShortcuts_h
#define __agtkImageFilteringShortcuts_h

#include <type_traits>

#include <itkRegionOfInterestImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkNumericTraits.h>

namespace agtk
{
//------------------------------------------------------------------------------
template <typename TLikeImage, typename TOutputImage = TLikeImage>
typename TOutputImage::Pointer createImageLike(const TLikeImage* imageLike, const typename TOutputImage::PixelType fillValue)
{
  static_assert(TLikeImage::ImageDimension == TOutputImage::ImageDimension,
                "Like image and output image dimensions must be equal");

  typename TOutputImage::Pointer image = TOutputImage::New();

  image->SetRegions(imageLike->GetLargestPossibleRegion());
  image->CopyInformation(imageLike);
  image->Allocate();
  image->FillBuffer(fillValue);

  return image;
}

//------------------------------------------------------------------------------
template <typename TImage>
typename TImage::Pointer cropImage(const TImage* image, const typename TImage::RegionType& region)
{
  typedef itk::RegionOfInterestImageFilter<TImage, TImage> ROI;
  typename ROI::Pointer roi = ROI::New();

  roi->SetInput(image);
  roi->SetRegionOfInterest(region);
  roi->Update();

  return roi->GetOutput();
}

//------------------------------------------------------------------------------
template <typename TImage>
typename TImage::Pointer pasteImage(const TImage* image, const TImage* regionImage, const typename TImage::IndexType& index)
{
  typedef itk::PasteImageFilter<TImage> Paste;
  typename Paste::Pointer paste = Paste::New();

  paste->SetDestinationImage(image);
  paste->SetSourceImage(regionImage);
  paste->SetSourceRegion(regionImage->GetLargestPossibleRegion());
  paste->SetDestinationIndex(index);

  paste->Update();

  return paste->GetOutput();
}

//------------------------------------------------------------------------------
template <typename TImage>
typename TImage::RegionType thresholdBoundingBoxRegion(const TImage* image, typename TImage::PixelType lowerThreshold, typename TImage::PixelType upperThreshold)
{
  typedef typename TImage::IndexType IndexType;
  typedef typename IndexType::IndexValueType IndexValueType;

  IndexType minIndex;
  IndexType maxIndex;

  minIndex.Fill(itk::NumericTraits<IndexValueType>::max());
  maxIndex.Fill(itk::NumericTraits<IndexValueType>::NonpositiveMin());

  itk::ImageRegionConstIteratorWithIndex<TImage> it(image, image->GetLargestPossibleRegion());

  it.GoToBegin();

  while (!it.IsAtEnd()) {
    if (it.Get() < lowerThreshold || it.Get() > upperThreshold) {
      ++it;
      continue;
    }

    IndexType index = it.GetIndex();

    if (index[0] < minIndex[0]) {
      minIndex[0] = index[0];
    }

    if (index[1] < minIndex[1]) {
      minIndex[1] = index[1];
    }

    if (index[2] < minIndex[2]) {
      minIndex[2] = index[2];
    }

    if (index[0] > maxIndex[0]) {
      maxIndex[0] = index[0];
    }

    if (index[1] > maxIndex[1]) {
      maxIndex[1] = index[1];
    }

    if (index[2] > maxIndex[2]) {
      maxIndex[2] = index[2];
    }

    ++it;
  }

  typename TImage::RegionType region;

  region.SetIndex(minIndex);
  region.SetUpperIndex(maxIndex);

  return region;
}
} // end of namespace agtk

#endif // __agtkImageFilteringShortcuts_h
