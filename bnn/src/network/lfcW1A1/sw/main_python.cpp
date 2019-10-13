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
 * @file main_python.c
 *
 * Host code for BNN, overlay LFC-Pynq, to manage parameter loading and 
 * classification (inference) on single or multiple images
 * 
 *
 *****************************************************************************/
 
#include "tiny_cnn/tiny_cnn.h"
#include "tiny_cnn/util/util.h"
#include <iostream>
#include <string.h>
#include <chrono>
#include "foldedmv-offload.h"
#include <random>
#include <algorithm>
#include "layers.h"

using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void makeNetwork(network<mse, adagrad> & nn) {
  nn
#ifdef OFFLOAD
    << offloaded_layer(28*28, 10, &FoldedMVOffload<ap_int<16>>, 0, 0)
#endif
  ;
}

extern "C" void load_parameters(const char* path) {
#include "config.h"
  FoldedMVInit("lfcW1A1-pynq");
  network<mse, adagrad> nn;
  makeNetwork(nn);
  cout << "Setting network weights and thresholds in accelerator..." << endl;
  FoldedMVLoadLayerMem(path, 0, L0_PE, L0_WMEM, L0_TMEM, L0_API);
  FoldedMVLoadLayerMem(path, 1, L1_PE, L1_WMEM, L1_TMEM, L1_API);
  FoldedMVLoadLayerMem(path, 2, L2_PE, L2_WMEM, L2_TMEM, L2_API);
  FoldedMVLoadLayerMem(path, 3, L3_PE, L3_WMEM, L3_TMEM, L3_API);
}

void random_fault(
	bool flip_word = false,
	int target_type = -1,
	int *target_layers = nullptr,
	unsigned int num_layers = 0
) {
#include "config.h"

	const layer_data layers = {
		{L0_PE,   L1_PE,   L2_PE,   L3_PE},
		{L0_WMEM, L1_WMEM, L2_WMEM, L3_WMEM},
		{L0_TMEM, L1_TMEM, L2_TMEM, L3_TMEM},
		{L0_SIMD, L1_SIMD, L2_SIMD, L3_SIMD},
		{L0_API,  L1_API,  L2_API,  L3_API},
		{L0_WPI,  L1_WPI,  L2_WPI,  L3_WPI},
		{16,      16,      16,      16}
	}

	std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> selection;
	if (target_layers) {
		std::vector<uint32_t> target_layers_vec(target_layers, target_layers + num_layers);
		selection = random_selection(layers, target_type, target_layers_vec);
	}
	else {
		selection = random_selection(layers, target_type);
	}

	inject_fault(
		std::get<0>(selection),
		std::get<1>(selection),
		std::get<2>(selection),
		std::get<3>(selection),
		std::get<4>(selection),
		flip_word
	);
}

extern "C" int inference(const char* path, int results[64], int number_class, float *usecPerImage) {
  std::vector<vec_t> test_images;
  std::vector<int> class_result;
  float usecPerImage_int;
  vec_t image;

  FoldedMVInit("lfcW1A1-pynq");
  network<mse, adagrad> nn;
  makeNetwork(nn);	
  parse_mnist_images(path, &test_images, -1.0, 1.0, 0, 0);
  image = test_images[0];
  class_result=testPrebinarized_nolabel(test_images, number_class, usecPerImage_int);

  if(results) {
    std::copy(class_result.begin(),class_result.end(), results);
  } 
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  return (std::distance(class_result.begin(),std::max_element(class_result.begin(), class_result.end())));
}

extern "C" int* inference_multiple(const char* path, int number_class, int *image_number, float *usecPerImage, unsigned int enable_detail = 0) {
  std::vector<vec_t> test_images;
  std::vector<int> all_result;
  float usecPerImage_int;
  int* result;

  FoldedMVInit("lfcW1A1-pynq");
  network<mse, adagrad> nn;
  makeNetwork(nn);	
  parse_mnist_images(path, &test_images, -1.0, 1.0, 0, 0);
  all_result=testPrebinarized_nolabel_multiple_images(test_images, number_class, usecPerImage_int);

  result = new int [all_result.size()];
  std::copy(all_result.begin(),all_result.end(), result);	
  if (image_number) {
    *image_number = all_result.size();
  }
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
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

  std::vector<vec_t> test_images;
  std::vector<int> all_result;
  float usecPerImage_int;
  int* result;

  FoldedMVInit("lfcW1A1-pynq");
  network<mse, adagrad> nn;
  makeNetwork(nn);	
  parse_mnist_images(path, &test_images, -1.0, 1.0, 0, 0);

	// Random device/generator
	std::random_device rd;
	std::mt19937 gen{rd()};

	// Used to determine what time in the process to inject a fault
	std::uniform_int_distribution<size_t> loop_dist{0, test_images.size() - 1};

	// Generate fault times
	std::vector<size_t> fault_indices;
	for (unsigned int i = 0; i < flip_count; ++i) {
		fault_indices.push_back(loop_dist(gen));
	}

	std::vector<vec_t> single_img;
	// Process images and inject faults
	for (size_t i = 0; i < test_images.size(); ++i) {
		single_img.clear();
		single_img.push_back(test_images[i]);

		// Inject a fault
		if (std::find(fault_indices.begin(), fault_indices.end(), i) != fault_indices.end()) {
			random_fault(flip_word != 0, target_type, target_layers, num_layers);
		}

		// Classify next image
		const std::vector<int> class_result = testPrebinarized_nolabel_multiple_images(single_img, number_class, usecPerImage_int);
		for (const auto result : class_result) {
			all_result.push_back(result);
		}
	}

  result = new int [all_result.size()];
  std::copy(all_result.begin(),all_result.end(), result);	
  if (image_number) {
    *image_number = all_result.size();
  }
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  return result;
}

extern "C" void free_results(int * result) {
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
  int class_inference = 0;
  int scores[64];

  load_parameters(argv[1]);
  class_inference = inference(argv[2], scores, atol(argv[3]), &execution_time);	

  cout << "Detected class " << class_inference << " in " << execution_time << " microseconds" << endl;	
  deinit();	
  if (class_inference != atol(argv[4])) {
    return 1;
  } else {
    return 0;
  }
}
