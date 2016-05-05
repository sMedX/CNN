#ifdef caffefication_EXPORTS
#  define CAFFEFICATION_EXPORT  __declspec(dllexport)
#else
#  define CAFFEFICATION_EXPORT
#endif

#define DECLARE_DLL_FUNCTION(ret, name, params)\
  CAFFEFICATION_EXPORT ret name params;\
  typedef ret(*name##FuncPtr)params;

#include <itkImage.h>

#include <agtkTypes.h>

// Forward declaration
namespace caffe
{
  template <typename T>
  class Net;
}

using namespace agtk;

extern "C" DECLARE_DLL_FUNCTION(bool, classify, (caffe::Net<float>* caffeNet, const std::string& preset, Int16Image3D::Pointer image16,
  UInt8Image3D::Pointer imageMask, Image3DRegion& region, int radiusXY, float spacingXY, int batchLength, int groupX,
  int groupY, int classCount, bool isRgb, OUT BinaryImage3D::Pointer& outImage))

extern "C" DECLARE_DLL_FUNCTION(void, loadNet, (const std::string& modelFile, const std::string& trainedFile, int deviceId, std::shared_ptr<caffe::Net<float>>& caffeNet))