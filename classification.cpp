#include <caffe/caffe.hpp>

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

//ITK
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTestingExtractSliceImageFilter.h"
#include "itkMetaImageIOFactory.h"
#include "itkOpenCVImageBridge.h"
#include <itkAddImageFilter.h>

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef unsigned short PixelType;
typedef itk::Image<PixelType, 3> ImageType3D;
typedef itk::Image<PixelType, 2> ImageType2D;

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier
{
public:
  Classifier(const string& model_file,
    const string& trained_file,
    const string& mean_file,
    const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
    std::vector<cv::Mat>* input_channels);

private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
  const string& trained_file,
  const string& mean_file,
  const string& label_file)
{
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  Caffe::set_phase(Caffe::TEST);

  //net_.reset(new Net<float>(model_file, TEST));
  net_.reset(new Net<float>(model_file));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
  const std::pair<float, int>& rhs)
{
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N)
{
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. TODO N hardly reset to 2 */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N)
{
  std::vector<float> output = Predict(img);
  //std::cout << img << " 1 \n";
  //std::cout << output[0] << " / "<< output[1] << "\n";

  /*
  //--
  std::vector<int> maxN = Argmax(output, N);
  std::cout << maxN.size() << " 2 \n";
  std::cout << output.size() << " 3 \n";
  */
  std::vector<Prediction> predictions;
  //--
  N = 2;
  for (int i = 0; i < N; ++i) {
    //--int idx = maxN[i];
    int idx = i;
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file)
{
  BlobProto blob_proto;
  //cv::Mat mean_ = cv::imread(mean_file);

  //return;

  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
  * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img)
{
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
    input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);


  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
  std::vector<cv::Mat>* input_channels)
{
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  //--sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;


  //cv::subtract(sample_float, mean_, sample_normalized);

  sample_normalized = sample_float;
  //std::cout << "[*]" << sample_normalized << "\n";

  /* This operation will write the separate BGR planes directly to the
  * input layer of the network because it is wrapped by the cv::Mat
  * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
    == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

template <typename TPixel>
ImageType2D::Pointer getTile(const itk::Image<TPixel, 3>* image, const typename itk::Image<TPixel, 3>::IndexType& index, int halfSize)
{
  typedef itk::Image<TPixel, 3> ImageType3D;

  typedef itk::Testing::ExtractSliceImageFilter<ImageType3D, ImageType2D> ExtractVolumeFilterType;

  auto extractVolumeFilter = ExtractVolumeFilterType::New();
  typename ImageType3D::RegionType region = image->GetLargestPossibleRegion();

  typename ImageType3D::SizeType size = {2 * halfSize, 2 * halfSize, 0};
  typename ImageType3D::IndexType start = {index[0] - halfSize + 1, index[1] - halfSize + 1, index[2]};

  typename ImageType3D::RegionType outputRegion;
  outputRegion.SetSize(size);
  outputRegion.SetIndex(start);

  extractVolumeFilter->SetInput(image);
  extractVolumeFilter->SetExtractionRegion(outputRegion);
  extractVolumeFilter->SetDirectionCollapseToGuess();
  extractVolumeFilter->Update();

  return extractVolumeFilter->GetOutput();
}

void testOn2DTile(int argc, char** argv)
{
  string model_file = argv[1];
  string trained_file = argv[2];
  string mean_file = argv[3];
  string label_file = argv[4];
  string test_list = argv[5];

  std::cout <<
    "model_file = " << model_file << std::endl <<
    "trained_file =" << trained_file << std::endl <<
    "mean_file = " << mean_file << std::endl <<
    "label_file =" << label_file << std::endl <<
    "test_list =" << test_list << std::endl;

  std::cout << "load classifier" << std::endl;
  ::google::InitGoogleLogging("log.txt");
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  //itk::MetaImageIOFactory::RegisterOneFactory();

  std::cout << "." << std::endl;

  int TP = 0, FN = 0, FP = 0, TN = 0;

  std::ifstream testListFile(test_list);
  string fileName;
  int target;
  std::cout << "classify" << std::endl;
  int n = 0;
  while (testListFile >> fileName >> target) {
    cv::Mat img = cv::imread(fileName);

    std::vector<Prediction> predictions = classifier.Classify(img);
 
    int output = predictions[1].second > predictions[0].second;
    if (target == 1) {
      if (output == 1) TP++;
      else FN++;
    }
    else {
      if (output == 1) FP++;
      else TN++;
    }
    if ((n + 1) % 100 == 0) {
      std::cout << n << std::endl;
    }
  }
  std::cout << "TP " << TP << std::endl;
  std::cout << "FN " << FN << std::endl;
  std::cout << "FP " << FP << std::endl;
  std::cout << "TN " << TN << std::endl;

  double sens, spec, voe, accuracy;

  sens = TP / (TP + FN);
  spec = TN / (TN + FP);
  voe = (1 - TP / (FP + TP + FN));
  accuracy = (TP + TN) / (FP + TP + FN + TN);

  std::cout << "sens " << sens << std::endl;
  std::cout << "spec " << spec << std::endl;
  std::cout << "voe " << voe << std::endl;
  std::cout << "accuracy " << accuracy << std::endl;

}
#define TEST

int main(int argc, char** argv)
{
#ifdef TEST
  testOn2DTile(argc, argv);
  return EXIT_SUCCESS;
#endif // TEST

    std::cout << "Usage-:\n\
    string logger_file = argv[0];\n\
    string model_file = argv[1];\n\
    string trained_file = argv[2];\n\
    string mean_file = argv[3];\n\
    string label_file = argv[4];\n\
    \n\
    string start_x_str = argv[5];\n\
    string start_y_str = argv[6];\n\
    string start_z_str = argv[7];\n\
    \n\
    string size_x_str = argv[8];\n\
    string size_y_str = argv[9];\n\
    string size_z_str = argv[10];\n\
    \n\
    string radiusXY = argv[11];\n\
    \n\
    string input_file = argv[12];\n\
    string mask_file = argv[13];\n\
    string output_file = argv[14];\n\
    \n\
    string radiusXY_str = argv[14];\n\
    string shift_str = argv[15]"
    << std::endl;

  string logger_file = argv[0];
  string model_file = argv[1];
  string trained_file = argv[2];
  string mean_file = argv[3];
  string label_file = argv[4];

  string start_x_str = argv[5];
  string start_y_str = argv[6];
  string start_z_str = argv[7];

  string size_x_str = argv[8];
  string size_y_str = argv[9];
  string size_z_str = argv[10];

  string input_file = argv[11];
  string mask_file = argv[12];
  string output_file = argv[13];

  //string radiusXY_str = argv[14];
  //string shift_str = argv[15];

  ImageType3D::IndexType start;
  start[0] = atoi(start_x_str.c_str());
  start[1] = atoi(start_y_str.c_str());
  start[2] = atoi(start_z_str.c_str());

  ImageType3D::SizeType size;
  size[0] = atoi(size_x_str.c_str());
  size[1] = atoi(size_y_str.c_str());
  size[2] = atoi(size_z_str.c_str());

  ImageType3D::RegionType region;
  region.SetIndex(start);
  region.SetSize(size);

  int radiusXY =  32; //atoi(radiusXY_str.c_str())
  int shift = 500; // atoi(shift_str.c_str());

  std::cout << "logger_file = " << argv[0] << std::endl <<
    "model_file = " << argv[1] << std::endl <<
    "trained_file =" << argv[2] << std::endl <<
    "mean_file = " << argv[3] << std::endl <<
    "label_file =" << argv[4] << std::endl <<
    "region = " << region << std::endl <<
    "radiusXY=" << radiusXY << std::endl <<
    "shift=" << shift << std::endl <<
    "input_file = " << argv[11] << std::endl <<
    "mask_file =" << argv[12] << std::endl <<
    "output_file =" << argv[13] << std::endl;
    
  std::cout << "load images" << std::endl;
  
  itk::MetaImageIOFactory::RegisterOneFactory();

  std::cout << "." << std::endl;

  typedef itk::ImageFileReader<ImageType3D> ReaderType;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(input_file);
  try {
    reader->Update();
  } catch (itk::ExceptionObject &excp) {
    std::cout << "Exception thrown while reading the image " << std::endl;
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "." << std::endl;

  ReaderType::Pointer readerMask = ReaderType::New();
  readerMask->SetFileName(mask_file);
  try {
    readerMask->Update();
  } catch (itk::ExceptionObject &excp) {
    std::cout << "Exception thrown while reading the mask " << std::endl;
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "." << std::endl;

  ImageType3D::ConstPointer image = reader->GetOutput();
  ImageType3D::ConstPointer imageMask = readerMask->GetOutput();

  //shift image
  if (shift != 0) {
    typedef itk::AddImageFilter <ImageType3D> AddImageFilterType;
    AddImageFilterType::Pointer addImageFilter = AddImageFilterType::New();
    addImageFilter->SetInput(image);
    addImageFilter->SetConstant2(shift);
    addImageFilter->Update();
    image = addImageFilter->GetOutput();
  }

  if (image->GetLargestPossibleRegion() != imageMask->GetLargestPossibleRegion()) {
    std::cout << "image->GetLargestPossibleRegion() != imageMask->GetLargestPossibleRegion() " << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "." << std::endl;

  //
  //region = image->GetLargestPossibleRegion();
  itk::ImageRegionConstIterator<ImageType3D> it(image, region);
  itk::ImageRegionConstIterator<ImageType3D> itMask(imageMask, region);
  
  std::cout << "." << std::endl;

  ImageType3D::Pointer outImage = ImageType3D::New();
  outImage->CopyInformation(image);
  outImage->SetRegions(image->GetLargestPossibleRegion());
  outImage->Allocate();
  outImage->FillBuffer(0);
  
  std::cout << "." << std::endl;

  int itCount = 0;
  int tumCount = 0;

  std::cout << "Calculating indices" << std::endl;

  vector<ImageType3D::IndexType> indices;
  
  //int c = 0;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++itMask) {
    //std::cout << it.GetIndex() << std::endl;
    if (itMask.Get() != 0) {
      //if (++c % 1000 == 0) { // debug. take each of 1000 element
        indices.push_back(it.GetIndex());
      //}
    }
  }

  const int totalCount = indices.size();

  std::cout << "Applying CNN in deploy config" << std::endl;

  ::google::InitGoogleLogging(logger_file.c_str());

  Classifier classifier(model_file, trained_file, mean_file, label_file);

  //#pragma omp parallel for private(img)
  for (int i = 0; i < totalCount; ++i) {
    auto tile = getTile<unsigned short>(image, indices[i], radiusXY);

    // need only when index is near with border
    //if (tile == nullptr) {
    //  continue;
    //}

    auto img = itk::OpenCVImageBridge::ITKImageToCVMat< ImageType2D >(tile);

    std::vector<Prediction> predictions = classifier.Classify(img);

    if (predictions[1].second > predictions[0].second) {
      //std::cout << predictions[0].first << " tum\n";
      tumCount++;
      outImage->SetPixel(indices[i], 1);
    } else {
      //std::cout << predictions[0].first << " NO tum\n";
      outImage->SetPixel(indices[i], 0);
    }

    ++itCount;

    if (itCount % 100 == 0) {
      std::cout << itCount << " / " << totalCount << "\n";
    }
  }

  std::cout << "tumors - " << tumCount << "\n";

  typedef itk::ImageFileWriter<ImageType3D>  writerType;
  writerType::Pointer writer = writerType::New();
  writer->SetFileName(output_file);
  writer->SetInput(outImage);
  try {
    writer->Update();
  } catch (itk::ExceptionObject &excp) {
    std::cout << "Exception thrown while writing " << std::endl;
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
