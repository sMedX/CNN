#ifndef __agtkImageSimilarityMetricsBase_hxx
#define __agtkImageSimilarityMetricsBase_hxx

namespace agtk
{
template <typename TFixedImage, typename TMovingImage>
void ImageSimilarityMetricsBase<TFixedImage, TMovingImage>::ValidateImages()
{
  // Get Fixed and Moving (segmented images)
  if (this->m_FixedImage.IsNull() || this->m_MovingImage.IsNull()) {
    itkExceptionMacro(<< "Need two input images");
  }

  if (this->m_FixedImage->GetLargestPossibleRegion().GetSize() !=
      this->m_MovingImage->GetLargestPossibleRegion().GetSize()) {
    itkExceptionMacro(<< "Image sizes must be equal");
  }

  if (this->m_FixedImage->GetOrigin() != this->m_MovingImage->GetOrigin()) {
    itkExceptionMacro(<< "Image origins must be equal");
  }

  if (this->m_FixedImage->GetSpacing() != this->m_MovingImage->GetSpacing()) {
    itkExceptionMacro(<< "Image spacings must be equal");
  }

  if (this->m_FixedImage->GetDirection() != this->m_MovingImage->GetDirection()) {
    itkExceptionMacro(<< "Image directions must be equal");
  }
}

template <typename TFixedImage, typename TMovingImage>
typename ImageSimilarityMetricsBase<TFixedImage, TMovingImage>::MeasureType
ImageSimilarityMetricsBase<TFixedImage, TMovingImage>::CalculateMetricValue(const MeasureType& numerator, const MeasureType& denominator)
{
  if (numerator == 0 && denominator == 0) {
    return metric_limits::quiet_NaN();
  }

  if (denominator == 0) {
    return metric_limits::infinity();
  }

  return numerator / denominator;
}
}

#endif // __agtkImageSimilarityMetricsBase_hxx
