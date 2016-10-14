/*
 * Following values are calculated on base of calculated Distance field:
 * 1) AverageSymmetricDistance
 * 2) RootMeanSquareSymDistance
 * 3) HausdorffSymmetricDistance
 */
#ifndef __agtkDistanceBasedMetrics_h
#define __agtkDistanceBasedMetrics_h

#include "agtkTypes.h"
#include "agtkImageSimilarityMetricsBase.h"

namespace agtk
{
template <typename TFixedImage, typename TMovingImage>
class DistanceBasedMetrics : public ImageSimilarityMetricsBase<TFixedImage, TMovingImage>
{
public:
  typedef DistanceBasedMetrics Self;
  typedef ImageSimilarityMetricsBase<TFixedImage, TMovingImage> Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  typedef itk::Image<float, TFixedImage::ImageDimension> DistanceMapImageType;
  typedef itk::Image<BinaryPixel, TFixedImage::ImageDimension> ContourImageType;

  typedef typename Superclass::MeasureType MeasureType;
  typedef typename Superclass::metric_limits metric_limits;

  itkTypeMacro(DistanceBasedMetrics, ImageSimilarityMetricsBase);
  itkNewMacro(Self);

public:
  void Evaluate() override;
  MetricsInfo GetMetricsInfo() const override;
  void PrintReport(std::ostream& os) const;

  itkSetMacro(Blurring, bool);
  itkGetMacro(Blurring, bool);
  itkBooleanMacro(Blurring);

  itkGetMacro(AverageDistance, MeasureType);
  itkGetMacro(RootMeanSquareDistance, MeasureType);
  itkGetMacro(HausdorffDistance, MeasureType);

protected:
  DistanceBasedMetrics();
  virtual ~DistanceBasedMetrics();

  template <typename TImageType>
  typename DistanceMapImageType::Pointer CalculateDistanceMap(const TImageType* image);

  typename ContourImageType::Pointer CalculateContour(const DistanceMapImageType* image);

  void ComputeDistances();

protected:
  bool m_Blurring;

  typename DistanceMapImageType::Pointer m_FixedDistanceMap;
  typename DistanceMapImageType::Pointer m_MovingDistanceMap;

  typename ContourImageType::Pointer m_FixedContour;
  typename ContourImageType::Pointer m_MovingContour;

  MeasureType m_SumDistance;
  MeasureType m_SumDistance2;
  MeasureType m_MaxDistance;

  MeasureType m_FixedContourLength;
  MeasureType m_MovingContourLength;

  MeasureType m_AverageDistance;
  MeasureType m_RootMeanSquareDistance;
  MeasureType m_HausdorffDistance;
};
}

#ifndef MU_MANUAL_INSTANTIATION
#include "agtkDistanceBasedMetrics.hxx"
#endif

#endif
