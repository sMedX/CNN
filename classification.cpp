
#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>

// ITK
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMetaImageIOFactory.h>
#include <itkNrrdImageIOFactory.h>
#include <itkPngImageIOFactory.h>

#include "C:/alex/agtk/Modules/Core/agtkResampling.h" //todo remove it or intergrate agtk

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"

#include <itkTestingExtractSliceImageFilter.h>
#include <regex>

template <typename TPixel>
agtk::FloatImage2D::Pointer getTile(const itk::Image<TPixel, 2>* image, const typename itk::Image<TPixel, 2>::IndexType& index, int halfSize)
{
  typedef itk::Image<TPixel, 2> ImageType2D;

  typedef itk::Testing::ExtractSliceImageFilter<ImageType2D, agtk::FloatImage2D> ExtractVolumeFilterType;

  auto extractVolumeFilter = ExtractVolumeFilterType::New();
  agtk::Image2DSize size = {2 * halfSize, 2 * halfSize};
  agtk::Image2DIndex start = {index[0] - halfSize + 1, index[1] - halfSize + 1};

  typename ImageType2D::RegionType outputRegion;
  outputRegion.SetSize(size);
  outputRegion.SetIndex(start);

  extractVolumeFilter->SetInput(image);
  extractVolumeFilter->SetExtractionRegion(outputRegion);
  extractVolumeFilter->SetDirectionCollapseToGuess();
  extractVolumeFilter->Update();

  return extractVolumeFilter->GetOutput();
}

// validate each image in file formatted by lines as 'path label'
// now it made combining to full-sized image from tiles. dbg porposes
int validateTileList(int argc, char** argv)
{
  using namespace caffe;

  string modelFile = argv[1];
  string trainedFile = argv[2];

  string listFile = argv[3];

  std::cout << "Applying CNN in deploy config" << std::endl;

  //Setting CPU or GPU
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);

  //get the net
  Net<float> caffe_test_net(modelFile, TEST);
  //get trained net
  caffe_test_net.CopyTrainedLayersFrom(trainedFile);

  std::cout << "load images" << std::endl;
  itk::PNGImageIOFactory::RegisterOneFactory();
  auto registeredIOs = itk::ObjectFactoryBase::CreateAllInstance("itkImageIOBase");

  typedef itk::ImageFileReader<agtk::Int16Image2D> ReaderType;

  agtk::BinaryImage2D::Pointer out = agtk::BinaryImage2D::New();
  agtk::Image2DSize size = {{575, 529}};
  out->SetRegions(size);
  out->Allocate();
  out->FillBuffer(128);

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);

  int classCount = 2;

  int batchLength = 1;
  int channels = 1;

  std::ifstream infile(listFile);
  std::string line;

  while (std::getline(infile, line)) {
    std::istringstream iss(line);

    std::string imageFile;
    int label;
    iss >> imageFile;
    iss >> label;
    //std::cout << "imageFile:" << imageFile << std::endl;

    //extract index
    std::regex ws_re("[\/._]+"); // whitespace
    std::vector<std::string> tokens;
    std::copy(std::sregex_token_iterator(imageFile.begin(), imageFile.end(), ws_re, -1),
      std::sregex_token_iterator(),
      std::back_inserter(tokens));

    agtk::Image2DIndex index = {atoi(tokens[6].c_str()), atoi(tokens[7].c_str())}; //6,7 is last and pre-last items in path
    std::cout << index << std::endl;

    //
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(imageFile);
    try {
      reader->Update();
    }
    catch (itk::ExceptionObject &excp) {
      std::cout << "Exception thrown while reading the image " << std::endl;
      std::cout << excp << std::endl;
      return EXIT_FAILURE;
    }
    agtk::Int16Image2D::Pointer image16 = reader->GetOutput();

    //std::cout << "cast image to float" << std::endl;
    typedef itk::CastImageFilter<agtk::Int16Image2D, agtk::FloatImage2D> Cast;
    auto cast = Cast::New();
    cast->SetInput(image16);
    cast->Update();
    agtk::FloatImage2D::Pointer image = cast->GetOutput();

    auto size = image->GetLargestPossibleRegion().GetSize();
    size_t tileSize = size[0] * size[1];

    //
    Blob<float>* blob = new Blob<float>(batchLength, channels, size[0], size[0]);

    float* dst = blob->mutable_cpu_data();
    auto tile = image->GetBufferPointer();

    memcpy(dst, tile, tileSize*sizeof(float));

    //fill the vector
    vector<Blob<float>*> bottom;
    bottom.push_back(blob);
    float type = 0.0;

    auto results = caffe_test_net.Forward(bottom, &type)[0]->cpu_data();
    delete blob;

    float max = 0;
    int max_i = 0;
    for (int j = 0; j < classCount; ++j) {
      float value = results[j];
      if (value > max) {
        max = value;
        max_i = j;
      }
    }

    int val = max_i;
    //std::cout << val << std::endl;

    //
    out->SetPixel(index, val * 255);
    //
    //  if (val == 1 && label == 1) {
    //    std::cout << "TP" << std::endl;
    //  }
    //  else if (val == 0 && label == 0) {
    //    std::cout << "TN" << std::endl;
    //  }
    //  else if (val == 1 && label == 0) {
    //    std::cout << "FP" << std::endl;
    //  }
    //  else // if (val == 0 && label == 1) {
    //    std::cout << "FN" << std::endl;
    //  }
  }

  //
  typedef itk::ImageFileWriter<agtk::BinaryImage2D>  writerType;
  writerType::Pointer writer = writerType::New();
  writer->SetFileName("tilesCombined.png");
  writer->SetInput(out);
  try {
    writer->Update();
  }
  catch (itk::ExceptionObject &excp) {
    std::cout << "Exception thrown while writing " << std::endl;
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
  }

  //
  return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
  //return validateTileList(argc, argv);

  using namespace caffe;

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

  string batchLengthStr = argv[12];

  string groupXStr = argv[13]; // interpret an area XxY as 1 unit
  string groupYStr = argv[14];

  string classCountStr = argv[15];

  string input_file = argv[16];
  string mask_file = argv[17];
  string output_file = argv[18];

  string deviceIdStr = argv[19];

  agtk::Image2DIndex start;
  start[0] = atoi(start_x_str.c_str());
  start[1] = atoi(start_y_str.c_str());
  start[2] = atoi(start_z_str.c_str());

  agtk::Image2DSize size;
  size[0] = atoi(size_x_str.c_str());
  size[1] = atoi(size_y_str.c_str());
  //size[2] = atoi(size_z_str.c_str());

  agtk::Image2DRegion region;
  region.SetIndex(start);
  region.SetSize(size);

  int radiusXY = atoi(radiusXY_str.c_str());
  float spacingXY = atof(spacingXY_str.c_str());
  int batchLength = atoi(batchLengthStr.c_str());
  int groupX = atoi(groupXStr.c_str());
  int groupY = atoi(groupYStr.c_str());
  int classCount = atoi(classCountStr.c_str());
  int deviceId = atoi(deviceIdStr.c_str());

  std::cout << "model_file = " << model_file << std::endl <<
    "trained_file =" << trained_file << std::endl <<
    "region = " << region << std::endl <<
    "radiusXY=" << radiusXY << std::endl <<
    "preset=" << preset << std::endl <<
    "spacingXY=" << spacingXY << std::endl <<
    "batchSize=" << batchLength << std::endl <<
    "groupX=" << groupX << std::endl <<
    "groupY=" << groupY << std::endl <<
    "classCount=" << classCount << std::endl <<
    "input_file = " << input_file << std::endl <<
    "mask_file =" << mask_file << std::endl <<
    "output_file =" << output_file << std::endl <<
    "deviceID =" << deviceId << std::endl;

  if (classCount != 2 && classCount != 4) {
    std::cout << "classCount must be 2 or 4";
  }

  std::cout << "load images" << std::endl;

  itk::MetaImageIOFactory::RegisterOneFactory();
  itk::NrrdImageIOFactory::RegisterOneFactory();
  itk::PNGImageIOFactory::RegisterOneFactory();

  std::cout << "nrrd factory registered" << std::endl;

  typedef std::list<itk::LightObject::Pointer> RegisteredObjectsContainerType;
  RegisteredObjectsContainerType registeredIOs = itk::ObjectFactoryBase::CreateAllInstance("itkImageIOBase");
  std::cout << "there are " << registeredIOs.size() << " IO objects available to the ImageFileReader." << std::endl;

  typedef itk::ImageFileReader<agtk::Int16Image2D> ReaderType;
  typedef itk::ImageFileReader<agtk::BinaryImage2D> BinaryReaderType;

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

  agtk::BinaryImage2D::Pointer imageMask;
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

  agtk::Int16Image2D::Pointer image16 = reader->GetOutput();

  std::cout << "preprocess images" << std::endl;
  std::cout << "shift, scale images" << std::endl;

  //if (preset == "pancreas") {
  //  const int shift = 190;// b
  //  const int squeeze = 2;// a

  //  // x' = (x + b)/a
  //  itk::ImageRegionIterator<agtk::Int16Image2D> it(image16, image16->GetLargestPossibleRegion());
  //  for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
  //    it.Set((it.Get() + shift) / squeeze);
  //  }
  //}
  //else if (preset == "livertumors") {
  //  const int shift = 40;

  //  // x'= x + shift
  //  itk::ImageRegionIterator<agtk::Int16Image2D> it(image16, image16->GetLargestPossibleRegion());
  //  for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
  //    it.Set(it.Get() + shift);
  //  }
  //}

  //if (spacingXY != 0) {
  //  std::cout << "resample image's axial slices" << std::endl;

  //  agtk::Image2DSpacing spacing;
  //  spacing[0] = spacingXY;
  //  spacing[1] = spacingXY;
  //  spacing[2] = image16->GetSpacing()[2];

  //  image16 = agtk::resampling(image16.GetPointer(), spacing);

  //  if (imageMask != nullptr) {
  //    imageMask = agtk::resamplingBinary(imageMask.GetPointer(), spacing);
  //  }
  //}

  std::cout << "cast image to float" << std::endl;
  typedef itk::CastImageFilter<agtk::Int16Image2D, agtk::FloatImage2D> Cast;
  auto cast = Cast::New();
  cast->SetInput(image16);
  cast->Update();
  agtk::FloatImage2D::Pointer image = cast->GetOutput();
  if (imageMask != nullptr) {
    if (image->GetLargestPossibleRegion() != imageMask->GetLargestPossibleRegion()) {
      std::cout << "image->GetLargestPossibleRegion() != imageMask->GetLargestPossibleRegion() " << std::endl;
      return EXIT_FAILURE;
    }
  }
  image16 = nullptr;

  std::cout << "Calculating indices" << std::endl;

  auto shrinkRegion = image->GetLargestPossibleRegion();
  const agtk::Image2DSize radius2D = {radiusXY, radiusXY};
  shrinkRegion.ShrinkByRadius(radius2D);
  region.Crop(shrinkRegion);

  vector<agtk::Image2DIndex> indices;
  std::cout << "region: " << region << std::endl;
  if (imageMask.IsNotNull()) {
    std::cout << "use mask" << std::endl;
    itk::ImageRegionConstIterator<agtk::BinaryImage2D> itMask(imageMask, region);

    for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
      if (itMask.Get() != 0) {
        //take central pixel of group
        const auto& index = itMask.GetIndex();
        if (index[0] % groupX == groupX / 2 && index[1] % groupY == groupY / 2) {
          indices.push_back(itMask.GetIndex());
        }
      }
    }
  }
  else {
    std::cout << "not use mask" << std::endl;
    itk::ImageRegionConstIterator<agtk::FloatImage2D> itMask(image, region);
    for (itMask.GoToBegin(); !itMask.IsAtEnd(); ++itMask) {
      //take central pixel of group
      auto& index = itMask.GetIndex();
      if (index[0] % groupX == groupX / 2 && index[1] % groupY == groupY / 2) {
        indices.push_back(itMask.GetIndex());
      }
    }
  }

  //reorder like 'zxy'
  //std::stable_sort(indices.begin(), indices.end(), [](agtk::Image2DIndex a, agtk::Image2DIndex b) // todo maybe use ordinal sort
  //{
  //  return a[2] != b[2] ? a[2] < b[2] : a[0] < b[0];
  //});
  const int totalCount = indices.size();

  std::cout << "total count:" << totalCount << std::endl;

  agtk::BinaryImage2D::Pointer outImage = agtk::BinaryImage2D::New();
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
  Caffe::SetDevice(deviceId);

  //get the net
  Net<float> caffe_test_net(model_file, TEST);
  //get trained net
  caffe_test_net.CopyTrainedLayersFrom(trained_file);

  //float* buffer = image->GetBufferPointer();

  const int channels = 1;
  // tile's properties
  const int height = 2 * radiusXY;
  const int width = 2 * radiusXY;
  const int tileSize = width*height;
  //const int lineSizeInBytes = 2 * radiusXY*sizeof(float);

  // image's properties
  //const auto& imageSize = image->GetLargestPossibleRegion().GetSize();
  //const int lineSize = imageSize[0];

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);

  for (int i = 0; i < totalCount / batchLength; ++i) { //todo fix last total%batch_size indices
    auto time0 = clock();
    std::cout << i << "th batch" << std::endl;

    Blob<float>* blob = new Blob<float>(batchLength, channels, height, width); // has been moved out from the loop

    for (int iTile = 0; iTile < batchLength; ++iTile) {
      const auto& index = indices[i*batchLength + iTile];
      ////if (index[0] == lastX) { // x same as before
      ////concat new line(s) to last tile
      ////} else {
      ////make tile from the scratch
      //const int xOffset = (index[0] - radiusXY + 1);
      //const int yOffsetPart = (index[1] - radiusXY + 1)* lineSize;
      //float* src = buffer + yOffsetPart + xOffset;

      const int tileOffset = iTile*tileSize;
      float* dst = blob->mutable_cpu_data() + tileOffset;

      //for (int iRow = 0; iRow < 2 * radiusXY; iRow++) { // try to compute offset by 1 vector command
      //  memcpy(dst, src, lineSizeInBytes);

      //  src += lineSize; // adjust yOffset
      //  dst += width; // adjust lineOffset
      //}
      auto tile = getTile(image.GetPointer(), index, radiusXY);
      //
      //std::string indexStr = std::to_string(index[0]) + "_" + std::to_string(index[1]);
      //std::string filename = "D:\\alex\\USI_tumor\\aug242015205152\\tiles\\" + indexStr + ".png";
      //typedef itk::ImageFileWriter<agtk::BinaryImage2D>  writerType;
      //writerType::Pointer writer = writerType::New();
      //writer->SetFileName(filename);
      //writer->SetInput(tile);
      //writer->Update();
      //
      memcpy(dst, tile->GetBufferPointer(), tileSize*sizeof(float));
    }

    //fill the vector
    vector<Blob<float>*> bottom;
    bottom.push_back(blob);
    float type = 0.0;

    auto time1 = clock();

    auto results = caffe_test_net.Forward(bottom, &type)[0]->cpu_data();
    delete blob;

    auto time2 = clock();
    //Here I can use the argmax layer, but for now I do a simple for :)
    for (int iTile = 0; iTile < batchLength; ++iTile) {
      auto& index = indices[i*batchLength + iTile];

      //float max = 0;
      //int max_i = 0;
      //for (int j = 0; j < classCount; ++j) {
      //  float value = results[classCount*iTile + j];
      //  if (value > max) {
      //    max = value;
      //    max_i = j;
      //  }
      //}

      int val = 255 * (results[2 * iTile + 1] - results[2 * iTile] + 1) / 2;

      //if (max_i == 1) {
      //  val = 255;
      //  tumCount++;
      //}

      //set group's area
      for (int k = -groupX / 2; k < groupX - groupX / 2; ++k) {
        for (int l = -groupY / 2; l < groupY - groupY / 2; ++l) {
          agtk::Image2DIndex index2 = {index[0] + k, index[1] + l};
          outImage->SetPixel(index2, val); // can be improved if only group 1x1 used
        }
      }

      //
      if (++itCount % 10000 == 0) {
        std::cout << itCount << " / " << totalCount << "\n";
      }
    }

    std::cout << "load data: " << static_cast<double>(time1 - time0) / CLOCKS_PER_SEC << std::endl;
    std::cout << "classify: " << static_cast<double>(time2 - time1) / CLOCKS_PER_SEC << std::endl;

  }

  std::cout << "tumors - " << tumCount << "\n";

  typedef itk::ImageFileWriter<agtk::BinaryImage2D>  writerType;
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
