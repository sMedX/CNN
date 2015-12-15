
#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>

// ITK
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTestingExtractSliceImageFilter.h>
#include <itkMetaImageIOFactory.h>

#include "C:/alex/agtk/Modules/Core/agtkResampling.h" //todo remove it or intergrate agtk

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"

using namespace caffe;
using namespace std;

template <typename TPixel>
agtk::UInt8Image2D::Pointer getTile(const itk::Image<TPixel, 3>* image, const typename itk::Image<TPixel, 3>::IndexType& index, int halfSize)
{
  typedef itk::Image<TPixel, 3> ImageType3D;

  typedef itk::Testing::ExtractSliceImageFilter<ImageType3D, agtk::UInt8Image2D> ExtractVolumeFilterType;

  auto extractVolumeFilter = ExtractVolumeFilterType::New();
  agtk::Image3DSize size = { 2 * halfSize, 2 * halfSize, 0 };
  agtk::Image3DIndex start = { index[0] - halfSize + 1, index[1] - halfSize + 1, index[2] };

  // check boundary
  agtk::Image3DSize allSize = image->GetLargestPossibleRegion().GetSize();
  agtk::Image3DIndex corner = start + size;

  typename ImageType3D::RegionType outputRegion;
  outputRegion.SetSize(size);
  outputRegion.SetIndex(start);

  extractVolumeFilter->SetInput(image);
  extractVolumeFilter->SetExtractionRegion(outputRegion);
  extractVolumeFilter->SetDirectionCollapseToGuess();
  extractVolumeFilter->Update();

  return extractVolumeFilter->GetOutput();
}

agtk::Image3DRegion getBinaryMaskBoundingBoxRegion(const agtk::BinaryImage3D* image)
{
  // TODO: This code can be parallelized
  agtk::Image3DIndex minIndex, maxIndex;

  minIndex.Fill(itk::NumericTraits<agtk::Image3DIndex::IndexValueType>::max());
  maxIndex.Fill(itk::NumericTraits<agtk::Image3DIndex::IndexValueType>::NonpositiveMin());

  itk::ImageRegionConstIteratorWithIndex<agtk::BinaryImage3D> it(image, image->GetLargestPossibleRegion());

  it.GoToBegin();

  while (!it.IsAtEnd()) {
    if (it.Get() != agtk::OUTSIDE_BINARY_VALUE) {
      agtk::Image3DIndex index = it.GetIndex();

      if (index[0] < minIndex[0])
        minIndex[0] = index[0];

      if (index[1] < minIndex[1])
        minIndex[1] = index[1];

      if (index[2] < minIndex[2])
        minIndex[2] = index[2];

      if (index[0] > maxIndex[0])
        maxIndex[0] = index[0];

      if (index[1] > maxIndex[1])
        maxIndex[1] = index[1];

      if (index[2] > maxIndex[2])
        maxIndex[2] = index[2];
    }

    ++it;
  }

  agtk::Image3DRegion region;

  region.SetIndex(minIndex);
  region.SetUpperIndex(maxIndex);

  return region;
}

//
//void testOn2DTiles(int argc, char** argv)
//{
//  string model_file = argv[1];
//  string trained_file = argv[2];
//  string mean_file = argv[3];
//  string label_file = argv[4];
//  string test_list = argv[5];
//
//  std::cout <<
//    "model_file = " << model_file << std::endl <<
//    "trained_file =" << trained_file << std::endl <<
//    "mean_file = " << mean_file << std::endl <<
//    "label_file =" << label_file << std::endl <<
//    "test_list =" << test_list << std::endl;
//
//  std::cout << "load classifier" << std::endl;
//  ::google::InitGoogleLogging("log.txt");
//  Classifier classifier(model_file, trained_file, mean_file, label_file);
//
//  std::cout << "." << std::endl;
//
//  int TP = 0, FN = 0, FP = 0, TN = 0;
//
//  std::ifstream testListFile(test_list);
//  string fileName;
//  int target;
//  std::cout << "classify" << std::endl;
//  int n = 0;
//  while (testListFile >> fileName >> target) {
//    cv::Mat img = cv::imread(fileName, -1);
//    if (img.empty()){
//      std::cout << "Unable to decode image " << fileName << std::endl;
//    }
//
//    std::vector<Prediction> predictions = classifier.Classify(img);
//
//    int output = predictions[0].first == "1";
//    if (target == 1) {
//      if (output == 1) TP++;
//      else FN++;
//    }
//    else {
//      if (output == 1) FP++;
//      else TN++;
//    }
//    
//    std::cout << "1: " << predictions[1].second << ", 0: " << predictions[0].second << std::endl;
//    if ((n++) % 100 == 0) {
//      std::cout << n - 1 << std::endl;
//    }
//  }
//  std::cout << "TP " << TP << std::endl;
//  std::cout << "FN " << FN << std::endl;
//  std::cout << "FP " << FP << std::endl;
//  std::cout << "TN " << TN << std::endl;
//
//  double sens, spec, voe, accuracy;
//
//  sens = TP / (TP + FN);
//  spec = TN / (TN + FP);
//  voe = (1 - TP / (FP + TP + FN));
//  accuracy = (TP + TN) / (FP + TP + FN + TN);
//
//  std::cout << "sens " << sens << std::endl;
//  std::cout << "spec " << spec << std::endl;
//  std::cout << "voe " << voe << std::endl;
//  std::cout << "accuracy " << accuracy << std::endl;
//
//}
//int oldmain(int argc, char** argv)
//{
//  std::cout << "class NEW\n";
//  if (argc < 4 || argc > 6) {
//    LOG(ERROR) << "deploy caffemodel img_file "
//      << "[CPU/GPU] [Device ID]";
//    return 1;
//  }
//  //Caffe::set_phase(Caffe::TEST);
//
//  //Setting CPU or GPU
//  if (argc >= 5 && strcmp(argv[4], "GPU") == 0) {
//    Caffe::set_mode(Caffe::GPU);
//    int device_id = 0;
//    if (argc == 6) {
//      device_id = atoi(argv[5]);
//    }
//    Caffe::SetDevice(device_id);
//    LOG(ERROR) << "Using GPU #" << device_id;
//  }
//  else {
//    LOG(ERROR) << "Using CPU";
//    Caffe::set_mode(Caffe::CPU);
//  }
//
//  //get the net
//  Net<float> caffe_test_net(argv[1], TEST);
//  //get trained net
//  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
//
//  //get datum
//  Datum datum;
//  if (!ReadImageToDatum(argv[3], 1, 64, 64, &datum, false)) {
//    LOG(ERROR) << "Error during file reading";
//  }
//  LOG(INFO) << "datum.channels() " << datum.channels();
//  //get the blob
//  Blob<float>* blob = new Blob<float>(1, datum.channels(), datum.height(), datum.width());
//
//  //get the blobproto
//  BlobProto blob_proto;
//  blob_proto.set_num(1);
//  blob_proto.set_channels(datum.channels());
//  blob_proto.set_height(datum.height());
//  blob_proto.set_width(datum.width());
//  const int data_size = datum.channels() * datum.height() * datum.width();
//  int size_in_datum = std::max<int>(datum.data().size(),
//    datum.float_data_size());
//  for (int i = 0; i < size_in_datum; ++i) {
//    blob_proto.add_data(0.);
//  }
//  const string& data = datum.data();
//  if (data.size() != 0) {
//    for (int i = 0; i < size_in_datum; ++i) {
//      //original was uint8_t, so it also work well, 
//      //but for the case of dicom data uint16_t is better :-)
//      //blob_proto.set_data(i, blob_proto.data(i) + (uint16_t)data[i]);
//      blob_proto.set_data(i, blob_proto.data(i) + (uint16_t)data[i]);
//    }
//  }
//
//  //set data into blob
//  blob->FromProto(blob_proto);
//
//  //fill the vector
//  vector<Blob<float>*> bottom;
//  bottom.push_back(blob);
//  float type = 0.0;
//
//  const vector<Blob<float>*>& result = caffe_test_net.Forward(bottom, &type);
//
//  //Here I can use the argmax layer, but for now I do a simple for :)
//  float max = 0;
//  float max_i = 0;
//  for (int i = 0; i <= 1; ++i) {
//    float value = result[0]->cpu_data()[i];
//    if (max < value){
//      max = value;
//      max_i = i;
//    }
//  }
//  LOG(ERROR) << "max: " << max << " i " << max_i;
//
//  return 0;
//
//}
//void testOn2DTile(int argc, char** argv)
//{
//  if (argc != 6) {
//    std::cerr << "Usage: " << argv[0]
//      << " deploy.prototxt network.caffemodel"
//      << " mean.binaryproto labels.txt img.jpg" << std::endl;
//    return;
//  }
//  std::cout << "1\n";
//
//  ::google::InitGoogleLogging(argv[0]);
//
//  string model_file = argv[1];
//  string trained_file = argv[2];
//  string mean_file = argv[3];
//  string label_file = argv[4];
//  std::cout << "2\n";
//
//  Classifier classifier(model_file, trained_file, mean_file, label_file);
//
//  string file = argv[5];
//
//  std::cout << "---------- Prediction for "
//    << file << " ----------" << std::endl;
//
//  cv::Mat img = cv::imread(file, -1);
//  CHECK(!img.empty()) << "Unable to decode image " << file;
//  std::time_t result = std::time(nullptr);
//  std::cout << std::asctime(std::localtime(&result)) << "\n";
//  std::vector<Prediction> predictions;
//  predictions = classifier.Classify(img);
//  result = std::time(nullptr);
//  std::cout << std::asctime(std::localtime(&result)) << "\n";
//
//  int output = predictions[0].first == "1";
//  std::cout << "output: " << output << std::endl;
//
//}

int main(int argc, char** argv) {

  //to test on list of 2d sample uncomment these lines
  //testOn2DTiles(argc,argv);
  //return EXIT_SUCCESS;

  std::cout << "Usage-:\n\
                 string model_file = argv[1];\n\
                   string trained_file = argv[2];\n\
                     \n\
                       string start_x_str = argv[3];\n\
                         string start_y_str = argv[4];\n\
                           string start_z_str = argv[5];\n\
                             \n\
                               string size_x_str = argv[6];\n\
                                 string size_y_str = argv[7];\n\
                                   string size_z_str = argv[8];\n\
                                     \n\
                                       string radiusXY_str = argv[9];\n\
                                         string preset = argv[10];\n\
                                           string spacingXY_str = argv[11]; \n\
                                             \n\
                                               string input_file = argv[12];\n\
                                                 string mask_file = argv[13];\n\
                                                   string output_file = argv[14];"
                                                   << std::endl;

  string model_file = argv[1];
  string trained_file = argv[2];

  string start_x_str = argv[3];
  string start_y_str = argv[4];
  string start_z_str = argv[5];

  string size_x_str = argv[6];
  string size_y_str = argv[7];
  string size_z_str = argv[8];

  string radiusXY_str = argv[9];
  string preset = argv[10];
  string spacingXY_str = argv[11];

  string input_file = argv[12];
  string mask_file = argv[13];
  string output_file = argv[14];

  agtk::Image3DIndex start;
  start[0] = atoi(start_x_str.c_str());
  start[1] = atoi(start_y_str.c_str());
  start[2] = atoi(start_z_str.c_str());

  agtk::Image3DSize size;
  size[0] = atoi(size_x_str.c_str());
  size[1] = atoi(size_y_str.c_str());
  size[2] = atoi(size_z_str.c_str());

  agtk::Image3DRegion region;
  region.SetIndex(start);
  region.SetSize(size);

  int radiusXY = atoi(radiusXY_str.c_str());
  int spacingXY = atoi(spacingXY_str.c_str());

  std::cout << "model_file = " << model_file << std::endl <<
    "trained_file =" << trained_file << std::endl <<
    "region = " << region << std::endl <<
    "radiusXY=" << radiusXY << std::endl <<
    "preset=" << preset << std::endl <<
    "spacingXY=" << spacingXY << std::endl <<
    "input_file = " << input_file << std::endl <<
    "mask_file =" << mask_file << std::endl <<
    "output_file =" << output_file << std::endl;

  std::cout << "load images" << std::endl;

  itk::MetaImageIOFactory::RegisterOneFactory();

  std::cout << "." << std::endl;

  typedef itk::ImageFileReader<agtk::Int16Image3D> ReaderType;
  typedef itk::ImageFileReader<agtk::BinaryImage3D> BinaryReaderType;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(input_file);
  try {
    reader->Update();
  }
  catch (itk::ExceptionObject &excp) {
    std::cout << "Exception thrown while reading the image " << std::endl;
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "." << std::endl;

  agtk::BinaryImage3D::Pointer imageMask;
  if (mask_file == "BOUNDING_BOX") {
    imageMask = nullptr;
  }
  else {
    BinaryReaderType::Pointer readerMask = BinaryReaderType::New();
    readerMask->SetFileName(mask_file);
    try {
      readerMask->Update();
    }
    catch (itk::ExceptionObject &excp) {
      std::cout << "Exception thrown while reading the mask " << std::endl;
      std::cout << excp << std::endl;
      return EXIT_FAILURE;
    }
    imageMask = readerMask->GetOutput();
  }
  std::cout << "." << std::endl;

  agtk::Int16Image3D::Pointer image16 = reader->GetOutput();

  //preprocess image - shift image
  agtk::FloatImage3D::Pointer image;

  if (preset == "pancreas") {
    const int shift = 190;// b
    const int squeeze = 2;// a

    // x' = (x + b)/a
    itk::ImageRegionIterator<agtk::Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
      it.Set((it.Get() + shift) / squeeze);
    }
  }
  else if (preset == "livertumors") {
    const int shift = 40;
    itk::ImageRegionIterator<agtk::Int16Image3D> it(image16, image16->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
      it.Set(it.Get() + shift);
    }
  }

  if (spacingXY != 0) { //resample image by axial slices
    agtk::Image3DSpacing spacing;
    spacing[0] = spacingXY;
    spacing[1] = spacingXY;
    spacing[2] = image->GetSpacing()[2];

    image = agtk::resampling(image.GetPointer(), spacing);

    if (imageMask != nullptr) {
      imageMask = agtk::resamplingBinary(imageMask.GetPointer(), spacing);
    }
  }

  typedef itk::CastImageFilter<agtk::Int16Image3D, agtk::FloatImage3D> Cast;
  auto cast = Cast::New();
  cast->SetInput(image16);
  cast->Update();
  image = cast->GetOutput();
  if (imageMask != nullptr) {
    if (image->GetLargestPossibleRegion() != imageMask->GetLargestPossibleRegion()) {
      std::cout << "image->GetLargestPossibleRegion() != imageMask->GetLargestPossibleRegion() " << std::endl;
      return EXIT_FAILURE;
    }
  }
  std::cout << "." << std::endl;

  //
  std::cout << "Calculating indices" << std::endl;

  auto shrinkRegion = image->GetLargestPossibleRegion();
  shrinkRegion.ShrinkByRadius(radiusXY);
  region.Crop(shrinkRegion);

  vector<agtk::Image3DIndex> indices;
  std::cout << "region: " << region << std::endl;
  if (imageMask.IsNotNull()) {
    std::cout << "use mask" << std::endl;
    itk::ImageRegionConstIterator<agtk::BinaryImage3D> itMask(imageMask, region);

    for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
      if (itMask.Get() != 0) {
        //exp
        auto& index = itMask.GetIndex();
        if (index[0] % 3 == 1 && index[1] % 3 == 1) {
          indices.push_back(itMask.GetIndex());
        }
      }
    }
  }
  else {
    std::cout << "not use mask" << std::endl;
    itk::ImageRegionConstIterator<agtk::FloatImage3D> itMask(image, region);
    for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
      //exp
      auto& index = itMask.GetIndex();
      if (index[0] % 3 == 1 && index[1] % 3 == 1) {
        indices.push_back(itMask.GetIndex());
      }
    }
  }
  const int totalCount = indices.size();

  std::cout << "total count:" << totalCount << std::endl;

  agtk::BinaryImage3D::Pointer outImage = agtk::BinaryImage3D::New();
  outImage->CopyInformation(image);
  outImage->SetRegions(image->GetLargestPossibleRegion());
  outImage->Allocate();
  outImage->FillBuffer(0);

  std::cout << "." << std::endl;

  int itCount = 0;
  int tumCount = 0;

  std::cout << "Applying CNN in deploy config" << std::endl;

  //Setting CPU or GPU
  Caffe::set_mode(Caffe::GPU);
  const int device_id = 1;
  Caffe::SetDevice(device_id);

  //get the net
  Net<float> caffe_test_net(model_file, TEST);
  //get trained net
  caffe_test_net.CopyTrainedLayersFrom(trained_file);

  const int channels = 1;
  const int height = 2 * radiusXY;
  const int width = 2 * radiusXY;
  const int data_size = height*width*channels;
  const int batchLength = 1024;

  //omp_set_nested(1);
  //omp_set_num_threads(4);
  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);

  //#pragma omp parallel for
  for (int i = 0; i < totalCount / batchLength; ++i) { //todo fix last total%batch_size indices
    auto time0 = clock();
    std::cout << i << "th batch" << std::endl;

    Blob<float>* blob = new Blob<float>(batchLength, channels, height, width);

    BlobProto blob_proto;
    blob_proto.set_num(batchLength);
    blob_proto.set_channels(blob->channels());
    blob_proto.set_height(blob->height());
    blob_proto.set_width(blob->width());
    //float* data = new float[batchLength * data_size];

    //#pragma omp parallel for
    for (int batch_i = 0; batch_i < batchLength; ++batch_i) {
      //auto time_0 = clock();
      auto tile = getTile(image.GetPointer(), indices[i*batchLength + batch_i], radiusXY);
      //auto time_1 = clock();

      auto dataPointer = tile->GetBufferPointer();

      for (int j = 0; j < data_size; ++j) {
        blob_proto.add_data(dataPointer[j]);
      }
      //auto time_2 = clock();
      //std::cout << "gettile: " << (double)(time_1 - time_0)*1000/CLOCKS_PER_SEC <<
      //  ". add to proto: " << (double)(time_2 - time_1)*1000 / CLOCKS_PER_SEC << std::endl;

      //std::cout << batch_i << std::endl;
    }
    blob->FromProto(blob_proto);
    //blob->data().get()->set_cpu_data(data);
    //fill the vector
    vector<Blob<float>*> bottom;
    bottom.push_back(blob);
    float type = 0.0;

    auto time1 = clock();

    auto results = caffe_test_net.Forward(bottom, &type)[0]->cpu_data();
    delete blob;
    //delete[] data;
    auto time2 = clock();
    //Here I can use the argmax layer, but for now I do a simple for :)
    for (int batch_i = 0; batch_i < batchLength; ++batch_i) {
      auto& index = indices[i*batchLength + batch_i];

      /* use it for multiclass problem
      float max = 0;
      float max_i = 0;
      for (int j = 0; j < 2; ++j) {
      float value = results->cpu_data()[batch_i + j];
      if (max < value){
      max = value;
      max_i = j;
      }
      }
      std::cout << "max: " << max << " i " << max_i << std::endl;
      */
      char val;
      if (results[batch_i * 2 + 1] > results[batch_i * 2]) {
        val = 1;
        tumCount++;
        //std::cout << "1" << std::endl;
      }
      else {
        val = 0;
        //std::cout << "0" << std::endl;
      }
      //outImage->SetPixel(index, val);
      //exp
      for (int k = -1; k < 2; ++k) {
        for (int l = -1; l < 2; ++l) {
          agtk::Image3DSize offset = { k, l, 0 };
          auto index2 = index + offset;
          outImage->SetPixel(index2, val);
        }
      }
      //
      if (++itCount % 1000 == 0) {
        std::cout << itCount << " / " << totalCount << "\n";
      }
    }
    auto time3 = clock();
    std::cout << "load data: " << (double)(time1 - time0) / CLOCKS_PER_SEC << std::endl;
    std::cout << "classify, delete data: " << (double)(time2 - time1) / CLOCKS_PER_SEC << std::endl;
    std::cout << "set data into image: " << (double)(time3 - time2) / CLOCKS_PER_SEC << std::endl;

  }

  std::cout << "tumors - " << tumCount << "\n";

  typedef itk::ImageFileWriter<agtk::BinaryImage3D>  writerType;
  writerType::Pointer writer = writerType::New();
  writer->SetFileName(output_file);
  writer->SetInput(outImage);
  try {
    writer->Update();
  }
  catch (itk::ExceptionObject &excp) {
    std::cout << "Exception thrown while writing " << std::endl;
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
