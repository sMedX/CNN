#include <itkConstantPadImageFilter.h>

#include "agtkResampling.h"

namespace
{
  using namespace agtk;

  BinaryImage3D::Pointer padImage(const BinaryImage3D* image, const itk::ImageBase<3>::SizeType& outputRegion)
  {
    typedef BinaryImage3D ImageType;
    typedef itk::ConstantPadImageFilter <ImageType, ImageType> ConstantPadImageFilterType;

    const ImageType::PixelType constantPixel = 0;

    auto padFilter = ConstantPadImageFilterType::New();
    padFilter->SetInput(image);
    padFilter->SetPadBound(outputRegion); // Calls SetPadLowerBound(region) and SetPadUpperBound(region)
    padFilter->SetConstant(constantPixel);
    padFilter->Update();

    return padFilter->GetOutput();
  }

  // Performs preprocessing with casting to uint8
  UInt8Image3D::Pointer smartCastImage(std::string& preset, Int16Image3D::Pointer& image16)
  {
    std::cout << "shift, sqeeze" << std::endl;

    if (preset == "pancreas") {
      const int shift = 190;
      const int squeeze = 2;

      // x' = (x + shift)/squeeze
      itk::ImageRegionIterator<Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set((it.Get() + shift) / squeeze);
      }
    } else if (preset == "livertumors") {
      const int shift = 40;

      // x' = x + shift
      itk::ImageRegionIterator<Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(it.Get() + shift);
      }
    }
    std::cout << "cast (truncate)" << std::endl;
    // force integer overflow
    UInt8Image3D::PixelType minValue = 0, maxValue = 255;

    itk::ImageRegionIterator<Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
      auto val = it.Get();
      if (val < minValue) {
        val = minValue;
      } else if (val > maxValue) {
        val = maxValue;
      }
      it.Set(val);
    }

    typedef itk::CastImageFilter<Int16Image3D, UInt8Image3D> Cast;
    auto cast = Cast::New();
    cast->SetInput(image16);
    cast->Update();
    return cast->GetOutput();
  }

  // Performs preprocessing befory cutting by tiles
  void preprocess(int radius, std::string& IN preset, float spacingXY, bool isRgb, Int16Image3D::Pointer& IN image16,
    UInt8Image3D::Pointer& IN OUT label1, UInt8Image3D::Pointer&IN OUT label2, UInt8Image3D::Pointer& IN OUT mask, UInt8Image3D::Pointer& IN OUT adaptive,
    UInt8Image3D::Pointer& OUT image)
  {
    image = smartCastImage(preset, image16);

    if (spacingXY != 0) { //resample image by axial slices
      std::cout << "resample" << std::endl;
      Image3DSpacing spacing;
      spacing[0] = spacingXY;
      spacing[1] = spacingXY;
      spacing[2] = image->GetSpacing()[2];

      image = resampling(image.GetPointer(), spacing);
      if (label1.IsNotNull()) {
        label1 = resamplingBinary(label1.GetPointer(), spacing);
      }
      if (label2.IsNotNull()) {
        label2 = resamplingBinary(label2.GetPointer(), spacing);
      }

      if (mask.IsNotNull()) {
        mask = resamplingBinary(mask.GetPointer(), spacing);
      }

      if (adaptive.IsNotNull()) {
        adaptive = resamplingBinary(adaptive.GetPointer(), spacing);
      }
    }

    const Image3DOffset radius3D = { radius, radius, isRgb ? 1 : 0 };
    const Image3DSize size3D = { radius3D[0], radius3D[1], radius3D[2] };
    std::cout << "padding by radius " << size3D << std::endl;

    image = padImage(image, size3D);
    if (label1.IsNotNull()) {
      label1 = padImage(label1.GetPointer(), size3D);
    }
    if (label2.IsNotNull()) {
      label2 = padImage(label2.GetPointer(), size3D);
    }

    if (mask.IsNotNull()) {
      mask = padImage(mask.GetPointer(), size3D);
    }

    if (adaptive.IsNotNull()) {
      adaptive = padImage(adaptive.GetPointer(), size3D);
    }
  }
}