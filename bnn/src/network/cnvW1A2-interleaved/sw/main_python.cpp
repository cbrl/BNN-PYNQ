/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file main_python.cpp
 *
 * Host code for CNV network using 1 bit weights and 2 activation. Functions to manage parameter loading and 
 * classification on single or multiple images are provided.
 * 
 *
 *****************************************************************************/
 
#include "tiny_cnn/tiny_cnn.h"
#include "tiny_cnn/util/util.h"
#include <iostream>
#include <string.h>
#include <chrono>
#include "foldedmv-offload.h"
#include <algorithm>
#include "faults.h"

using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void makeNetwork(network<mse, adagrad> & nn) {
  nn
#ifdef OFFLOAD
    << chaninterleave_layer<identity>(3, 32*32, false)
    << offloaded_layer(3*32*32, 10, &FixedFoldedMVOffload<8, 1, ap_int<16>>, 0xdeadbeef, 0)
#endif
  ;
}

extern "C" void load_parameters(const char* path) {
#include "config.h"
  FoldedMVInit("cnvW1A2-pynq");
  network<mse, adagrad> nn;
  makeNetwork(nn);
  cout << "Setting network weights and thresholds in accelerator..." << endl;

	InterleavedLoadArgs<L0_SIMD * L0_WPI, 24> layer0_args(false, true);
	InterleavedLoadArgs<L1_SIMD * L1_WPI, 16> layer1_args(false, true);
	InterleavedLoadArgs<L2_SIMD * L2_WPI, 16> layer2_args(false, true);
	InterleavedLoadArgs<L3_SIMD * L3_WPI, 16> layer3_args(false, true);
	InterleavedLoadArgs<L4_SIMD * L4_WPI, 16> layer4_args(false, true);
	InterleavedLoadArgs<L5_SIMD * L5_WPI, 16> layer5_args(false, true);
	InterleavedLoadArgs<L6_SIMD * L6_WPI, 16> layer6_args(false, true);
	InterleavedLoadArgs<L7_SIMD * L7_WPI, 16> layer7_args(false, true);
	InterleavedLoadArgs<L8_SIMD * L8_WPI,  0> layer8_args(false, false);

	FoldedMVLoadInterleavedLayerMem(path, 0, L0_PE, L0_WMEM, L0_TMEM, L0_API, 0, layer0_args);
	FoldedMVLoadInterleavedLayerMem(path, 1, L1_PE, L1_WMEM, L1_TMEM, L1_API, 0, layer1_args);
	FoldedMVLoadInterleavedLayerMem(path, 2, L2_PE, L2_WMEM, L2_TMEM, L2_API, 0, layer2_args);
	FoldedMVLoadInterleavedLayerMem(path, 3, L3_PE, L3_WMEM, L3_TMEM, L3_API, 0, layer3_args);
	FoldedMVLoadInterleavedLayerMem(path, 4, L4_PE, L4_WMEM, L4_TMEM, L4_API, 0, layer4_args);
	FoldedMVLoadInterleavedLayerMem(path, 5, L5_PE, L5_WMEM, L5_TMEM, L5_API, 0, layer5_args);
	FoldedMVLoadInterleavedLayerMem(path, 6, L6_PE, L6_WMEM, L6_TMEM, L6_API, 0, layer6_args);
	FoldedMVLoadInterleavedLayerMem(path, 7, L7_PE, L7_WMEM, L7_TMEM, L7_API, 0, layer7_args);
	FoldedMVLoadInterleavedLayerMem(path, 8, L8_PE, L8_WMEM, L8_TMEM, 0,      0, layer8_args);
}

RandomFaultArgs make_random_fault_args(
	int flip_word,
	int target,
	int *target_layers,
	unsigned int num_layers
) {
#include "config.h"

	static const CNVTopology topology = {
		{1,       1,       1,       1,       1,       1,       1,       1,       1},
		{1,       1,       1,       1,       1,       1,       1,       1,       1},
		{L0_PE,   L1_PE,   L2_PE,   L3_PE,   L4_PE,   L5_PE,   L6_PE,   L7_PE,   L8_PE},
		{L0_WMEM, L1_WMEM, L2_WMEM, L3_WMEM, L4_WMEM, L5_WMEM, L6_WMEM, L7_WMEM, L8_WMEM},
		{L0_TMEM, L1_TMEM, L2_TMEM, L3_TMEM, L4_TMEM, L5_TMEM, L6_TMEM, L7_TMEM, L8_TMEM},
		{L0_SIMD, L1_SIMD, L2_SIMD, L3_SIMD, L4_SIMD, L5_SIMD, L6_SIMD, L7_SIMD, L8_SIMD},
		{L0_API,  L1_API,  L2_API,  L3_API,  L4_API,  L5_API,  L6_API,  L7_API,  L8_API},
		{L0_WPI,  L1_WPI,  L2_WPI,  L3_WPI,  L4_WPI,  L5_WPI,  L6_WPI,  L7_WPI,  L8_WPI},
		{24,      16,      16,      16,      16,      16,      16,      16,      0}
	};

	// Target Type
	TargetType target_type;
	if (target < 0) target_type = TargetType::Any;
	else if (target == 0) target_type = TargetType::Weights;
	else target_type = TargetType::Activations;

	// Make layers vector
	std::vector<uint32_t> layers;
	if (target_layers) layers = std::vector<uint32_t>{target_layers, target_layers + num_layers};

	// Injection func
	std::function<void(const NetworkTopology&, TargetType, bool, uint32_t, uint32_t)> func{inject_fault<CNVTopology>};
	
  return RandomFaultArgs{topology, layers, target_type, flip_word != 0, func};
}

extern "C" int inference(const char* path, int results[64], int number_class, float* usecPerImage) {
  std::vector<label_t> test_labels;
  std::vector<vec_t> test_images;
  std::vector<int> class_result;
  float usecPerImage_int;	

  FoldedMVInit("cnvW1A2-pynq");
  network<mse, adagrad> nn;
  makeNetwork(nn);
  parse_cifar10(path, &test_images, &test_labels, -1.0, 1.0, 0, 0);
  class_result=testPrebuiltCIFAR10_from_image<8, 16, ap_int<16>>(test_images, number_class, usecPerImage_int);

  if(results) {
    std::copy(class_result.begin(),class_result.end(), results);
  }
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  return (std::distance(class_result.begin(),std::max_element(class_result.begin(), class_result.end())));
}

extern "C" int* inference_multiple(const char* path, int number_class, int* image_number, float* usecPerImage, int enable_detail = 0) {
  std::vector<int> detailed_results;
  std::vector<label_t> test_labels;
  std::vector<vec_t> test_images;
  std::vector<int> all_result;
  float usecPerImage_int;	
  int * result;

  FoldedMVInit("cnvW1A2-pynq");	
  network<mse, adagrad> nn;	
  makeNetwork(nn);	
  parse_cifar10(path, &test_images, &test_labels, -1.0, 1.0, 0, 0);	
  all_result=testPrebuiltCIFAR10_multiple_images<8, 16, ap_int<16>>(test_images, number_class, detailed_results, usecPerImage_int);

  if (image_number) {
    *image_number = all_result.size();
  }
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  if (enable_detail) {
    result = new int [detailed_results.size()];
    std::copy(detailed_results.begin(),detailed_results.end(), result);
  } else {
    result = new int [all_result.size()];
    std::copy(all_result.begin(),all_result.end(), result);
  } 
  return result;
}

extern "C" int* inference_multiple_with_faults(
	const char* path,
	int number_class,
	int *image_number,
	float *usecPerImage,
	unsigned int flip_count,
	int flip_word = 0,
	int target_type = -1,
	int *target_layers = nullptr,
	unsigned int num_layers = 0
) {
	std::vector<int> all_result;
	std::vector<int> detailed_results;
	std::vector<label_t> test_labels;
	std::vector<vec_t> test_images;
	float usecPerImage_int;
	int * result;

	FoldedMVInit("cnvW1A2-pynq");
	network<mse, adagrad> nn;
	makeNetwork(nn);
	parse_cifar10(path, &test_images, &test_labels, -1.0, 1.0, 0, 0);

	const RandomFaultArgs fault_args = make_random_fault_args(flip_word, target_type, target_layers, num_layers);
  
	auto classification_func = make_faulty_classification_func(
		fault_args,
		test_images.size(),
		[&](uint32_t image_idx) {
			std::vector<vec_t> single_img = {test_images[image_idx]};

			// Classify next image
			const std::vector<int> class_result = testPrebuiltCIFAR10_multiple_images<8, 16, ap_int<16>>(single_img, number_class, detailed_results, usecPerImage_int);
			for (const auto result : class_result) {
				all_result.push_back(result);
			}
		}
	);

	// Classify images and inject faults
	classification_func(flip_count);

	if (image_number) {
	  *image_number = all_result.size();
	}
	if (usecPerImage) {
	  *usecPerImage = usecPerImage_int;
	}
	result = new int [all_result.size()];
	std::copy(all_result.begin(),all_result.end(), result);
	return result;
}

extern "C" void free_results(int* result) {
  delete[] result;
}

extern "C" void deinit() {
  FoldedMVDeinit();
}

extern "C" int main(int argc, char** argv) {
  if (argc != 5) {
    cout << "4 parameters are needed: " << endl;
    cout << "1 - folder for the binarized weights (binparam-***) - full path " << endl;
    cout << "2 - path to image to be classified" << endl;
    cout << "3 - number of classes in the dataset" << endl;
    cout << "4 - expected result" << endl;
    return 1;
  }

  float execution_time = 0;
  int class_inference;
  int scores[64];

  load_parameters(argv[1]);
  class_inference = inference(argv[2], scores, atol(argv[3]), &execution_time);

  cout << "Detected class " << class_inference << endl;
  cout << "in " << execution_time << " microseconds" << endl;
  deinit();

  if (class_inference != atol(argv[4])) {
    return 1;
  } else {
    return 0;
  }
}
