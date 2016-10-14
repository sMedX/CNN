#ifndef __agtkConfusionMatrixBasedMetrics_hxx
#define __agtkConfusionMatrixBasedMetrics_hxx

#include <string>

#include <itkImageRegionConstIterator.h>

#include "agtkConfusionMatrixBasedMetrics.h"

namespace agtk
{
template <typename TFixedImage, typename TMovingImage>
ConfusionMatrixBasedMetrics<TFixedImage, TMovingImage>::ConfusionMatrixBasedMetrics()
{
  m_TruePositiveRate = metric_limits::quiet_NaN();
  m_TrueNegativeRate = metric_limits::quiet_NaN();
  m_PositivePredictiveRate = metric_limits::quiet_NaN();
  m_NegativePredictiveRate = metric_limits::quiet_NaN();
  m_AccuracyValue = metric_limits::quiet_NaN();
  m_JaccardCoefficient = metric_limits::quiet_NaN();
  m_VolumeOverlapError = metric_limits::quiet_NaN();
  m_RelativeVolumeDifference = metric_limits::quiet_NaN();
  m_DiceCoefficient = metric_limits::quiet_NaN();
}

template <typename TFixedImage, typename TMovingImage>
ConfusionMatrixBasedMetrics<TFixedImage, TMovingImage>::~ConfusionMatrixBasedMetrics()
{
}

template <typename TFixedImage, typename TMovingImage>
void ConfusionMatrixBasedMetrics<TFixedImage, TMovingImage>::CalculateMatrix()
{
  typename TFixedImage::ConstPointer fixedImage = this->m_FixedImage;
  typename TMovingImage::ConstPointer movingImage = this->m_MovingImage;

  // Get count of true negatives and false positives
  m_TrueNegative = 0;
  m_FalsePositive = 0;

  // Get count of true negatives and false positives
  m_TruePositive = 0;
  m_FalseNegative = 0;

  // Handle special case where inputs are zeros
  // Define iterators
  typedef itk::ImageRegionConstIterator<TFixedImage> FixedIteratorType;
  typedef itk::ImageRegionConstIterator<TMovingImage> MovingIteratorType;

  FixedIteratorType fixedIt(fixedImage, fixedImage->GetLargestPossibleRegion());
  MovingIteratorType movingIt(movingImage, movingImage->GetLargestPossibleRegion());

  fixedIt.GoToBegin();
  movingIt.GoToBegin();

  while (!fixedIt.IsAtEnd() && !movingIt.IsAtEnd()) {
    bool a = (fixedIt.Get() != 0);
    bool b = (movingIt.Get() != 0);

    if (!a && !b) {
      m_TrueNegative++;
    }

    if (!a && b) {
      m_FalsePositive++;
    }

    if (a && b) {
      m_TruePositive++;
    }

    if (a && !b) {
      m_FalseNegative++;
    }

    ++fixedIt;
    ++movingIt;
  }
}

template <typename TFixedImage, typename TMovingImage>
void ConfusionMatrixBasedMetrics<TFixedImage, TMovingImage>::Evaluate()
{
  this->ValidateImages();
  CalculateMatrix();

  MeasureType numA = m_TruePositive + m_FalseNegative;
  MeasureType numB = m_FalsePositive + m_TruePositive;
  MeasureType numUnion = m_TruePositive + m_FalsePositive + m_FalseNegative;

  // TPR
  m_TruePositiveRate = this->CalculateMetricValue(m_TruePositive, m_TruePositive + m_FalseNegative);

  // TNR
  m_TrueNegativeRate = this->CalculateMetricValue(m_TrueNegative, m_TrueNegative + m_FalsePositive);

  // Precision
  m_PositivePredictiveRate = this->CalculateMetricValue(m_TruePositive, m_TruePositive + m_FalsePositive);

  // NPV
  m_NegativePredictiveRate = this->CalculateMetricValue(m_TrueNegative, m_TrueNegative + m_FalseNegative);

  // Accuracy = (TP+TN)/(TP+TN+FP+FN)
  m_AccuracyValue = this->CalculateMetricValue(m_TruePositive + m_TrueNegative, m_TrueNegative + m_FalseNegative + m_TruePositive + m_FalsePositive);

  // Jaccard coefficient according to article
  m_JaccardCoefficient = this->CalculateMetricValue(m_TruePositive, numUnion);

  m_VolumeOverlapError = 1.0 - this->CalculateMetricValue(m_TruePositive, numUnion);
  m_RelativeVolumeDifference = this->CalculateMetricValue(numA - numB, numA);

  // Overlap or similarity coeff is intersect / average
  m_DiceCoefficient = 2.0 * this->CalculateMetricValue(m_TruePositive, numA + numB);
}

//----------------------------------------------------------------------------
template <typename TFixedImage, typename TMovingImage>
MetricsInfo ConfusionMatrixBasedMetrics<TFixedImage, TMovingImage>::GetMetricsInfo() const
{
  MetricsInfo metricsInfo;

  metricsInfo.push_back(MetricInfo(m_TruePositive, "TP", "True positive"));
  metricsInfo.push_back(MetricInfo(m_FalseNegative, "FN", "False negative"));
  metricsInfo.push_back(MetricInfo(m_FalsePositive, "FP", "False positive"));
  metricsInfo.push_back(MetricInfo(m_TrueNegative, "TN", "True negative"));

  metricsInfo.push_back(MetricInfo(m_TruePositiveRate, "TPR", "True positive rate"));
  metricsInfo.push_back(MetricInfo(m_TrueNegativeRate, "TNR", "True negative rate"));
  metricsInfo.push_back(MetricInfo(m_PositivePredictiveRate, "Precision", "Positive predictive rate"));
  metricsInfo.push_back(MetricInfo(m_NegativePredictiveRate, "NPR", "Negative predictive rate"));
  metricsInfo.push_back(MetricInfo(m_AccuracyValue, "Accuracy", "Accuracy"));

  metricsInfo.push_back(MetricInfo(m_RelativeVolumeDifference, "RVD", "Relative volume difference"));
  metricsInfo.push_back(MetricInfo(m_VolumeOverlapError, "VOE", "Volume overlap error"));
  metricsInfo.push_back(MetricInfo(m_DiceCoefficient, "Dice", "Dice coefficient"));
  metricsInfo.push_back(MetricInfo(m_JaccardCoefficient, "Jaccard", "Jaccard coefficient"));

  return metricsInfo;
}

template <typename TFixedImage, typename TMovingImage>
void ConfusionMatrixBasedMetrics<TFixedImage, TMovingImage>::PrintReport(std::ostream& os) const
{
  std::string indent = "    ";

  os << "Calculated Metric Values:" << std::endl;

  os << indent << "True positive (TP)                        = " << m_TruePositive << std::endl;
  os << indent << "False negative (FN)                       = " << m_FalseNegative << std::endl;
  os << indent << "False positive (FP)                       = " << m_FalsePositive << std::endl;
  os << indent << "True negative (TN)                        = " << m_TrueNegative << std::endl;
  os << std::endl;
  os << indent << "True positive rate (sensitivity)          = " << m_TruePositiveRate << std::endl;
  os << indent << "True negative rate (specificity)          = " << m_TrueNegativeRate << std::endl;
  os << indent << "Positive predictive rate (precision, PPR) = " << m_PositivePredictiveRate << std::endl;
  os << indent << "Negative predictive rate (NPR)            = " << m_NegativePredictiveRate << std::endl;
  os << indent << "Accuracy value                            = " << m_AccuracyValue << std::endl;
  os << std::endl;
  os << indent << "Relative volume difference (RVD)          = " << m_RelativeVolumeDifference << std::endl;
  os << indent << "Volume overlap error (VOE)                = " << m_VolumeOverlapError << std::endl;
  os << indent << "Dice coefficient                          = " << m_DiceCoefficient << std::endl;
  os << indent << "Jaccard coefficient                       = " << m_JaccardCoefficient << std::endl;
}
}
#endif
