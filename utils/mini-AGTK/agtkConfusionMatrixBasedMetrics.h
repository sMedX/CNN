/*
 * https://en.wikipedia.org/wiki/Confusion_matrix
 *
 * Following values are calculated on base of ConfusionMatrix:
 * 1) Volume Overlap Error
 * 2) Relative Volume Difference
 * 3) Dice Coefficient
 * 4) Jaccard Coefficient
 * 5) Sensitivity
 * 6) Specificity
 * 7) Precision
 * 8) Accuracy
 */
#ifndef __agtkConfusionMatrixBasedMetrics_h
#define __agtkConfusionMatrixBasedMetrics_h

#include "agtkImageSimilarityMetricsBase.h"

namespace agtk
{
template <typename TFixedImage, typename TMovingImage>
class ConfusionMatrixBasedMetrics : public ImageSimilarityMetricsBase<TFixedImage, TMovingImage>
{
public:
  /** Standard class typedefs. */
  typedef ConfusionMatrixBasedMetrics Self;
  typedef ImageSimilarityMetricsBase<TFixedImage, TMovingImage> Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  typedef typename Superclass::MeasureType MeasureType;
  typedef typename Superclass::metric_limits metric_limits;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConfusionMatrixBasedMetrics, ImageSimilarityMetricsBase);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

public:
  void Evaluate() override;
  MetricsInfo GetMetricsInfo() const override;
  void PrintReport(std::ostream& os) const;

public:
  // All metrics GET methods
  itkGetMacro(TruePositive, MeasureType);
  itkGetMacro(FalseNegative, MeasureType);
  itkGetMacro(FalsePositive, MeasureType);
  itkGetMacro(TrueNegative, MeasureType);

  itkGetMacro(TruePositiveRate, MeasureType);
  itkGetMacro(TrueNegativeRate, MeasureType);
  itkGetMacro(PositivePredictiveRate, MeasureType);
  itkGetMacro(NegativePredictiveRate, MeasureType);
  itkGetMacro(AccuracyValue, MeasureType);

  itkGetMacro(RelativeVolumeDifference, MeasureType);
  itkGetMacro(DiceCoefficient, MeasureType);
  itkGetMacro(JaccardCoefficient, MeasureType);
  itkGetMacro(VolumeOverlapError, MeasureType);

protected:
  ConfusionMatrixBasedMetrics();
  virtual ~ConfusionMatrixBasedMetrics();

  void CalculateMatrix();

protected:
  // Get count of true negatives and false positives
  MeasureType m_TrueNegative;
  MeasureType m_FalsePositive;

  // Get count of true negatives and false positives
  MeasureType m_TruePositive;
  MeasureType m_FalseNegative;

  MeasureType m_TruePositiveRate;
  MeasureType m_TrueNegativeRate;
  MeasureType m_PositivePredictiveRate;
  MeasureType m_NegativePredictiveRate;
  MeasureType m_AccuracyValue;

  MeasureType m_RelativeVolumeDifference;
  MeasureType m_DiceCoefficient;
  MeasureType m_JaccardCoefficient;
  MeasureType m_VolumeOverlapError;
};
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "agtkConfusionMatrixBasedMetrics.hxx"
#endif

#endif // __agtkConfusionMatrixBasedMetrics_h
