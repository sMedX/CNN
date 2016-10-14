#ifndef __agtkDistanceBasedMetrics_hxx
#define __agtkDistanceBasedMetrics_hxx

#include <cmath>
#include <string>

#include <itkImageRegionConstIterator.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkZeroCrossingImageFilter.h>

#include "agtkDistanceBasedMetrics.h"

namespace agtk
{
template <typename TFixedImage, typename TMovingImage>
DistanceBasedMetrics<TFixedImage, TMovingImage>::DistanceBasedMetrics()
{
  m_Blurring = false;

  m_AverageDistance = metric_limits::quiet_NaN();
  m_RootMeanSquareDistance = metric_limits::quiet_NaN();
  m_HausdorffDistance = metric_limits::quiet_NaN();
}

template <typename TFixedImage, typename TMovingImage>
DistanceBasedMetrics<TFixedImage, TMovingImage>::~DistanceBasedMetrics()
{
}

template <typename TFixedImage, typename TMovingImage>
void DistanceBasedMetrics<TFixedImage, TMovingImage>::ComputeDistances()
{
  typedef itk::ImageRegionConstIterator<DistanceMapImageType> DistanceMapIterator;
  typedef itk::ImageRegionConstIterator<ContourImageType> ContourIterator;

  DistanceMapIterator fixedDistanceMapIterator(m_FixedDistanceMap, m_FixedDistanceMap->GetLargestPossibleRegion());
  DistanceMapIterator movingDistanceMapIterator(m_MovingDistanceMap, m_MovingDistanceMap->GetLargestPossibleRegion());

  ContourIterator fixedContourIterator(m_FixedContour, m_FixedContour->GetLargestPossibleRegion());
  ContourIterator movingContourIterator(m_MovingContour, m_MovingContour->GetLargestPossibleRegion());

  fixedDistanceMapIterator.GoToBegin();
  movingDistanceMapIterator.GoToBegin();
  fixedContourIterator.GoToBegin();
  movingContourIterator.GoToBegin();

  m_SumDistance = 0;
  m_SumDistance2 = 0;
  m_MaxDistance = 0;

  m_FixedContourLength = 0;
  m_MovingContourLength = 0;

  while (!fixedDistanceMapIterator.IsAtEnd()) {
    if (fixedContourIterator.Get() > 0) {
      double dist = movingDistanceMapIterator.Get();

      m_SumDistance += std::abs(dist);
      m_SumDistance2 += dist * dist;
      m_MaxDistance = std::max(m_MaxDistance, dist);

      m_FixedContourLength++;
    }

    if (movingContourIterator.Get() > 0) {
      double dist = fixedDistanceMapIterator.Get();

      m_SumDistance += std::abs(dist);
      m_SumDistance2 += dist * dist;
      m_MaxDistance = std::max(m_MaxDistance, dist);

      m_MovingContourLength++;
    }

    ++fixedDistanceMapIterator;
    ++movingDistanceMapIterator;
    ++fixedContourIterator;
    ++movingContourIterator;
  }
}

template <typename TFixedImage, typename TMovingImage>
void DistanceBasedMetrics<TFixedImage, TMovingImage>::Evaluate()
{
  this->ValidateImages();

  m_FixedDistanceMap = CalculateDistanceMap<TFixedImage>(this->m_FixedImage);
  m_MovingDistanceMap = CalculateDistanceMap<TMovingImage>(this->m_MovingImage);

  m_FixedContour = CalculateContour(m_FixedDistanceMap);
  m_MovingContour = CalculateContour(m_MovingDistanceMap);

  ComputeDistances();

  if (m_FixedContourLength == 0 || m_MovingContourLength == 0) {
    m_AverageDistance = metric_limits::quiet_NaN();
    m_RootMeanSquareDistance = metric_limits::quiet_NaN();
    m_HausdorffDistance = metric_limits::quiet_NaN();
  }
  else {
    m_AverageDistance = this->CalculateMetricValue(m_SumDistance, m_FixedContourLength + m_MovingContourLength);
    m_RootMeanSquareDistance = std::sqrt(this->CalculateMetricValue(m_SumDistance2, m_FixedContourLength + m_MovingContourLength));
    m_HausdorffDistance = m_MaxDistance;
  }
}

template <typename TFixedImage, typename TMovingImage>
template <typename TImageType>
typename DistanceBasedMetrics<TFixedImage, TMovingImage>::DistanceMapImageType::Pointer
DistanceBasedMetrics<TFixedImage, TMovingImage>::CalculateDistanceMap(const TImageType* image)
{
  typedef itk::SignedMaurerDistanceMapImageFilter<TImageType, DistanceMapImageType> DistanceMapFilterType;
  typename DistanceMapFilterType::Pointer distanceMapFilter = DistanceMapFilterType::New();

  distanceMapFilter->SetInput(image);
  distanceMapFilter->InsideIsPositiveOff();
  distanceMapFilter->SquaredDistanceOff();
  distanceMapFilter->UseImageSpacingOn();
  distanceMapFilter->Update();

  return distanceMapFilter->GetOutput();
}

template <typename TFixedImage, typename TMovingImage>
typename DistanceBasedMetrics<TFixedImage, TMovingImage>::ContourImageType::Pointer
DistanceBasedMetrics<TFixedImage, TMovingImage>::CalculateContour(const DistanceMapImageType* image)
{
  typename DistanceMapImageType::ConstPointer distanceImage = image;

  if (m_Blurring) {
    typename DistanceMapImageType::SpacingType spacing = distanceImage->GetSpacing();
    double minSpacing = spacing[0];

    for (unsigned int dim = 0; dim < DistanceMapImageType::ImageDimension; dim++) {
      if (spacing[dim] < minSpacing) {
        minSpacing = spacing[dim];
      }
    }

    typedef itk::DiscreteGaussianImageFilter<DistanceMapImageType, DistanceMapImageType> DiscreteGaussianImageFilter;
    typename DiscreteGaussianImageFilter::Pointer blur = DiscreteGaussianImageFilter::New();

    blur->SetInput(distanceImage);
    blur->SetVariance(1.5 * minSpacing);
    blur->Update();

    distanceImage = blur->GetOutput();
  }

  typedef itk::ZeroCrossingImageFilter<DistanceMapImageType, ContourImageType> ZeroCrossingImageFilter;
  typename ZeroCrossingImageFilter::Pointer zeroCrossing = ZeroCrossingImageFilter::New();

  zeroCrossing->SetInput(distanceImage);
  zeroCrossing->SetBackgroundValue(0);
  zeroCrossing->SetForegroundValue(1);
  zeroCrossing->Update();

  return zeroCrossing->GetOutput();
}

//----------------------------------------------------------------------------
template <typename TFixedImage, typename TMovingImage>
MetricsInfo agtk::DistanceBasedMetrics<TFixedImage, TMovingImage>::GetMetricsInfo() const
{
  MetricsInfo metricsInfo;

  metricsInfo.push_back(MetricInfo(m_AverageDistance, "ASD", "Average distance", MetricUnits::Millimeters));
  metricsInfo.push_back(MetricInfo(m_RootMeanSquareDistance, "RMSD", "Root mean square distance", MetricUnits::Millimeters));
  metricsInfo.push_back(MetricInfo(m_HausdorffDistance, "Hausdorff", "Hausdorff distance", MetricUnits::Millimeters));

  return metricsInfo;
}

template <typename TFixedImage, typename TMovingImage>
void DistanceBasedMetrics<TFixedImage, TMovingImage>::PrintReport(std::ostream& os) const
{
  std::string indent = "    ";

  os << "Calculated Distance Metric Values:" << std::endl;
  os << indent << "Average distance (ASD)                    = " << m_AverageDistance << std::endl;
  os << indent << "Root mean square distance (RMSD)          = " << m_RootMeanSquareDistance << std::endl;
  os << indent << "Hausdorff distance (Maximal)              = " << m_HausdorffDistance << std::endl;
}
}
#endif
