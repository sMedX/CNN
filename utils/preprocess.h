#include <itkConstantPadImageFilter.h>

#include "agtkResampling.h"

namespace
{
  using namespace agtk;

  agtk::BinaryImage3D::Pointer padImage(const agtk::BinaryImage3D* image, const itk::ImageBase<3>::SizeType& outputRegion)
  {
    typedef agtk::BinaryImage3D ImageType;
    typedef itk::ConstantPadImageFilter <ImageType, ImageType> ConstantPadImageFilterType;

    const ImageType::PixelType constantPixel = 0;

    auto padFilter = ConstantPadImageFilterType::New();
    padFilter->SetInput(image);
    padFilter->SetPadBound(outputRegion); // Calls SetPadLowerBound(region) and SetPadUpperBound(region)
    padFilter->SetConstant(constantPixel);
    padFilter->Update();

    return padFilter->GetOutput();
  }

  void preprocess(int radius, std::string& preset, agtk::Image2DSpacing& spacingXY, bool isRgb, itk::Image<short, 3>::Pointer& image16, itk::Image<unsigned char, 3>::Pointer& label1, itk::Image<unsigned char, 3>::Pointer& label2, itk::Image<unsigned char, 3>::Pointer& mask, itk::Image<unsigned char, 3>::Pointer& adaptive, bool isAdaptiveClasses, itk::Image<unsigned char, 3>::Pointer& image)
  {
    std::cout << "shift, sqeeze" << std::endl;

    if (preset == "pancreas") {
      const int shift = 190;
      const int squeeze = 2;

      // x' = (x + shift)/squeeze
      itk::ImageRegionIterator<agtk::Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set((it.Get() + shift) / squeeze);
      }
    } else if (preset == "livertumors") {
      const int shift = 40;

      // x' = x + shift
      itk::ImageRegionIterator<agtk::Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(it.Get() + shift);
      }
    }
    std::cout << "cast (truncate)" << std::endl;
    // force integer overflow
    agtk::UInt8Image3D::PixelType minValue = 0, maxValue = 255;

    itk::ImageRegionIterator<agtk::Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
      auto val = it.Get();
      if (val < minValue) {
        val = minValue;
      } else if (val > maxValue) {
        val = maxValue;
      }
      it.Set(val);
    }

    typedef itk::CastImageFilter<agtk::Int16Image3D, agtk::UInt8Image3D> Cast;
    auto cast = Cast::New();
    cast->SetInput(image16);
    cast->Update();
    image = cast->GetOutput();

    if (spacingXY[0] != 0) { //resample image by axial slices
      std::cout << "resample" << std::endl;
      agtk::Image3DSpacing spacing;
      spacing[0] = spacingXY[0];
      spacing[1] = spacingXY[1];
      spacing[2] = image->GetSpacing()[2];

      image = resampling(image.GetPointer(), spacing);
      label1 = resamplingBinary(label1.GetPointer(), spacing);
      if (label2.IsNotNull()) {
        label2 = resamplingBinary(label2.GetPointer(), spacing);
      }

      if (mask.IsNotNull()) {
        mask = resamplingBinary(mask.GetPointer(), spacing);
      }

      if (isAdaptiveClasses) {
        adaptive = resamplingBinary(adaptive.GetPointer(), spacing);
      }
    }

    const agtk::Image3DOffset radius3D = { radius, radius, isRgb ? 1 : 0 };
    const agtk::Image3DSize size3D = { radius3D[0], radius3D[1], radius3D[2] };
    std::cout << "padding by radius " << size3D << std::endl;

    image = padImage(image, size3D);
    label1 = padImage(label1.GetPointer(), size3D);
    if (label2.IsNotNull()) {
      label2 = padImage(label2.GetPointer(), size3D);
    }

    if (mask.IsNotNull()) {
      mask = padImage(mask.GetPointer(), size3D);
    }

    if (isAdaptiveClasses) {
      adaptive = padImage(adaptive.GetPointer(), size3D);
    }
  }
}