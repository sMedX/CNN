/*
Fatless concurrent Caffe-based CNN classifier prototype
+Batch classification in single Net.Forward call
+omp parallel input data filling

Main purpose - CT images object/background segmentation.
Aimed to bi-class classification through file-list of grey-level uint8 tiles.

Author: Dr. Artem Nikonorov, <artniko@gmail.com>, www.AlignedResearchGroup.com
Date: 04 Dec 2015

Example usage:
class_new_batch.exe pan_03_deploy.prototxt _03__iter_160000.caffemodel GPU 0 16 _test_tiles.txt
Lines of the tiles list file:
D:/tiles/Negative/n7615-264_213_246.png 0
D:/tiles/Positive/n6166-225_249_188.png 1

*/

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>
#include "../../include/caffe/caffe.hpp"
#include "../../include/caffe/util/io.hpp"
#include "../../include/caffe/blob.hpp"

#define TILE_SIZE 64

using namespace caffe;
using namespace std;

int main(int argc, char** argv) {

	cout << "Batch Caffe-based CNN classifier\n";

	::google::InitGoogleLogging(argv[0]);
	if (argc < 6) {
		LOG(ERROR) << "deploy caffemodel CPU/GPU Device_ID batch_size img_list_file";
		return 1;
	}

	//Setting CPU or GPU
	if (strcmp(argv[3], "GPU") == 0) {
		Caffe::set_mode(Caffe::GPU);
		int device_id = 0;
		device_id = atoi(argv[4]);
		Caffe::SetDevice(device_id);
		LOG(ERROR) << "Using GPU #" << device_id;
	}
	else {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}

	//get the net
	Net<float> caffeNet(argv[1], TEST);
	//get trained net
	caffeNet.CopyTrainedLayersFrom(argv[2]);

	int batchSize = atoi(argv[5]);

	//get file and label lists
	string fileListName = argv[6];
	ifstream  fin(fileListName);
	string fileStr;
	vector<string> fileNames;
	vector<string> classLabels;
	int ind = 0;
	while (getline(fin, fileStr) && (ind < batchSize))
	{
		fileNames.push_back(fileStr.substr(0, fileStr.length() - 2));
		classLabels.push_back(fileStr.substr(fileStr.length() - 1));
		ind++;
	}

	//get the blob
	Blob<float>* blob = new Blob<float>(batchSize, 1, TILE_SIZE, TILE_SIZE);
	float max = 0;
	float max_i = 0;
	vector<Blob<float>*> bottom;

	//get the blobproto
	BlobProto blobProto;
	blobProto.set_num(batchSize);
	blobProto.set_channels(1);
	blobProto.set_height(64);
	blobProto.set_width(64);
	for (int i = 0; i < TILE_SIZE * TILE_SIZE * batchSize; i++) {
		blobProto.add_data(0.);
	}

	LOG(ERROR) << "Start" ;
#pragma omp parallel for
	for (int batchStep = 0; batchStep < batchSize; batchStep++) {
		//LOG(ERROR) << "loop start: " << batchStep;

		Datum datum;
		if (!ReadImageToDatum(fileNames[batchStep], 1, TILE_SIZE, TILE_SIZE, false, &datum)) {
			LOG(ERROR) << "Error during file reading";
		}

		const int data_size = datum.channels() * datum.height() * datum.width();
		int size_in_datum = std::max<int>(datum.data().size(), datum.float_data_size());

		//fill the blobproto
		const string& data = datum.data();
		if (data.size() != 0) {
			for (int i = 0; i < size_in_datum; ++i) {
				blobProto.set_data(batchStep * TILE_SIZE * TILE_SIZE + i, blobProto.data(batchStep * TILE_SIZE * TILE_SIZE + i) + (uint8_t)data[i]);
			}
		}

		//LOG(ERROR) << "loop end: " << batchStep;
	}//omp parallel for

	//set data into blob
	blob->FromProto(blobProto);
	//fill the net input vector
	bottom.push_back(blob);

	float type = 0.0;
	LOG(ERROR) << "Net Forward Start";
	const vector<Blob<float>*>& result = caffeNet.Forward(bottom, &type);
	LOG(ERROR) << "Net Forward End";
	LOG(ERROR) << "Result.count " << result[0]->count();
	//Possible speedup - try to use ForwardPrefilled()
	const vector<Blob<float>*>& result1 = caffeNet.ForwardPrefilled();

	LOG(ERROR) << "Result labeling output:";
	//process the net output
	int right = 0;
	int wrong = 0;
	for (int i = 0; i < batchSize; i++) {
		string classLabel;
		if (result[0]->cpu_data()[i * 2 + 1] > result[0]->cpu_data()[i * 2])
			classLabel = "1";
		else
			classLabel = "0";
		if (classLabel[0] == classLabels[i][0])
			right++;
		else
			wrong++;

		cout << classLabel << " : ";
	}
	cout << endl;

	LOG(ERROR) << "Tests count right/wrong: " << right << "/" << wrong;

	return 0;
}