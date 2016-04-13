#ifdef caffefication_EXPORTS
#  define CAFFEFICATION_EXPORT /*extern "C"*/ __declspec(dllexport)
#else
#  define CAFFEFICATION_EXPORT __declspec(dllimport)     //TODO
#endif

#define DECLARE_RUNTIME_FUNCTION(ret, name, params)\
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

DECLARE_RUNTIME_FUNCTION(int, classify, (caffe::Net<float>* caffeNet, std::string& preset, Int16Image3D::Pointer image16,
  UInt8Image3D::Pointer imageMask, Image3DRegion& region, int radiusXY, float spacingXY, int batchLength, int groupX,
  int groupY, int classCount, bool isRgb, OUT BinaryImage3D::Pointer& outImage))

DECLARE_RUNTIME_FUNCTION(void, loadNet, (std::string& modelFile, std::string trainedFile, int deviceId, OUT std::shared_ptr<caffe::Net<float>>& caffeNet))