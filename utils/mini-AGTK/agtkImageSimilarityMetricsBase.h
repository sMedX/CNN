#ifndef __agtkImageSimilarityMetricsBase_h
#define __agtkImageSimilarityMetricsBase_h

#include <limits>

#include <itkImageToImageMetric.h>

#include "agtkMetricInfo.h"

namespace agtk
{
template <typename TFixedImage, typename TMovingImage>
class ImageSimilarityMetricsBase : public itk::ImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  typedef ImageSimilarityMetricsBase Self;
  typedef itk::ImageToImageMetric<TFixedImage, TMovingImage> Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  typedef typename Superclass::MeasureType MeasureType;
  typedef typename Superclass::ParametersType ParametersType;
  typedef typename Superclass::DerivativeType DerivativeType;

  typedef std::numeric_limits<MeasureType> metric_limits;

  itkTypeMacro(ImageSimilarityMetricsBase, itk::ImageToImageMetric);

public:
  virtual void Evaluate() = 0;
  virtual MetricsInfo GetMetricsInfo() const = 0;

  virtual void PrintReport(std::ostream& os) const
  {
    itkExceptionMacro(<< "Not implemented");
  };

  virtual MeasureType GetValue(const ParametersType& p) const
  {
    itkExceptionMacro(<< "Not implemented");
  }

  virtual void GetDerivative(const ParametersType& p, DerivativeType& dp) const
  {
    itkExceptionMacro(<< "Not implemented");
  }

  virtual void GetValueAndDerivative(const ParametersType& p, MeasureType& v, DerivativeType& dp) const
  {
    itkExceptionMacro(<< "Not implemented");
  }

protected:
  ImageSimilarityMetricsBase() {};
  virtual ~ImageSimilarityMetricsBase() {};

  virtual void ValidateImages();
  MeasureType CalculateMetricValue(const MeasureType& numerator, const MeasureType& denominator);
};
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "agtkImageSimilarityMetricsBase.hxx"
#endif

#endif // __agtkImageSimilarityMetricsBase_h
