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
 * @file foldedmv-offload.h
 *
 * Library of functions for host code and managing HW offload
 * 
 *
 *****************************************************************************/

#pragma once
#include <string>
#include <iostream>
#include <bitset>
#include "tiny_cnn/tiny_cnn.h"
#include "ap_int.h"
#include "faults.h"

using namespace std;

typedef unsigned long long ExtMemWord;

const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord)*8;

#ifndef VIRTUAL
  #define INPUT_BUF_ENTRIES     3840000
  #define OUTPUT_BUF_ENTRIES    160000
#else
  #define INPUT_BUF_ENTRIES		8192
  #define OUTPUT_BUF_ENTRIES	1024
#endif

#define FOLDEDMV_INPUT_PADCHAR  0

void FoldedMVOffloadBinarized(const ExtMemWord * in, 
                              ExtMemWord * out,
                              const unsigned int inBufWords, 
                              const unsigned int outBufWords, 
                              const unsigned int numImages);

void FoldedMVInit(const char * attachName);

void FoldedMVDeinit();

void FoldedMVLoadLayerMem(std::string dir, 
                          unsigned int layerNo,
                          unsigned int peCount, 
                          unsigned int linesWMem, 
                          unsigned int linesTMem, 
                          unsigned int numThresh,
                          unsigned int numModules = 0);
void FoldedMVLoadWeightPE(ifstream& wf, unsigned int layerNo, unsigned int pe, unsigned int linesWMem, unsigned int numModules);
void FoldedMVLoadThreshPE(ifstream& tf, unsigned int layerNo, unsigned int pe, unsigned int linesTMem, unsigned int cntThresh, unsigned int numModules);

ExtMemWord FoldedMVMemRead(unsigned int targetLayer, 
                           unsigned int targetMem, 
                           unsigned int targetInd, 
                           unsigned int targetThresh,
                           int targetModule = -1);

void FoldedMVMemSet(unsigned int targetLayer, 
                    unsigned int targetMem, 
                    unsigned int targetInd, 
                    unsigned int targetThresh, 
                    ExtMemWord val,
                    int targetModule = -1);

std::vector<int> testPrebinarized_nolabel_multiple_images(std::vector<tiny_cnn::vec_t> & imgs, 
                                                          const unsigned int labelBits, 
                                                          float &usecPerImage);

std::vector<int> testPrebinarized_nolabel(std::vector<tiny_cnn::vec_t> & imgs, 
                                          const unsigned int labelBits, 
                                          float &usecPerImage);

void testPrebinarized(std::vector<tiny_cnn::vec_t> & imgs, 
                      std::vector<tiny_cnn::label_t> & labels, 
                      const unsigned int labelBits);

void binarizeAndPack(const tiny_cnn::vec_t & in, 
                     ExtMemWord * out, 
                     unsigned int inBufSize=INPUT_BUF_ENTRIES);

void unpackAndDebinarize(const ExtMemWord * in, tiny_cnn::vec_t &out);

unsigned int paddedSize(unsigned int in, unsigned int padTo);

std::string getBNNRoot();

template<typename LowPrecType>
void copyFromLowPrecBuffer(void * buf, tiny_cnn::vec_t & out) {
  LowPrecType * lpbuf = (LowPrecType *) buf;
  for(unsigned int i = 0; i < out.size(); i++) {
    out[i] = (tiny_cnn::float_t) lpbuf[i];
  }
}

template<unsigned int inWidth, unsigned int SIMDWidth>
void quantiseAndPack(const tiny_cnn::vec_t & in, ExtMemWord * out, unsigned int inBufSize=INPUT_BUF_ENTRIES) {
  if((in.size() * inWidth) > (inBufSize * bitsPerExtMemWord)) {
    throw "Not enough space in input buffer";
  }
  // first, fill the target buffer with padding data
  memset(out, 0, inBufSize * sizeof(ExtMemWord));
  ExtMemWord tmpv[bitsPerExtMemWord / inWidth];
  // now pack each quantised value as required.
  for(unsigned int i=0; i < in.size(); i++) {
    ap_fixed<inWidth, 1, AP_RND, AP_SAT> fxdValue = in[i];
    ap_uint<inWidth> uValue = *reinterpret_cast<ap_uint<inWidth> *>(&fxdValue); // Interpret the fixed value as an integer.
    ExtMemWord v = ((ExtMemWord)uValue & (~(ExtMemWord)0 >> (bitsPerExtMemWord - inWidth))); // Zero all bits except for the (bitsPerExtMemWord - inWidth) least significant bits.
    out[i / (bitsPerExtMemWord / inWidth)] |= (v << inWidth*(i % (bitsPerExtMemWord / inWidth)));
  }
}

inline void inject_fault_impl(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, int targetModule, unsigned int bit_pos, uint8_t word_size) {
  uint64_t val;
  if (targetModule >= 0) {
    val = FoldedMVMemRead(targetLayer, targetMem, targetInd, targetThresh, targetModule);
  }
  else {
    val = FoldedMVMemRead(targetLayer, targetMem, targetInd, targetThresh);
  }

  uint64_t flip_mask = 0;
  for (int i = 0; i < word_size; ++i) {
	flip_mask |= (uint64_t{1} << i);
  }
  val ^= (flip_mask << ((bit_pos/word_size)*word_size)); //aligns bit_pos to a multiple of word_size

  if (targetModule >= 0) {
    FoldedMVMemSet(targetLayer, targetMem, targetInd, targetThresh, val, targetModule);
  }
  else {
    FoldedMVMemSet(targetLayer, targetMem, targetInd, targetThresh, val);
  }
}

template<typename TopologyT>
void inject_fault(const NetworkTopology& abs_topology, TargetType target_type, uint8_t word_size, uint32_t layer, uint32_t bit) {
  const auto& topology = static_cast<const TopologyT&>(abs_topology);

  int module;
  unsigned int mem;
  unsigned int ind;
  unsigned int thresh;
  
  if ((target_type == TargetType::Weights) && (topology.weight_modules[layer] > 1)) {
    const uint32_t module_size = topology.weight_layer_bits[layer] / topology.weight_modules[layer];
    module = bit / module_size;
    bit    = bit % module_size;
  }
  else if ((target_type == TargetType::Activations) && (topology.activation_modules[layer] > 1)) {
    const uint32_t module_size = topology.activation_layer_bits[layer] / topology.activation_modules[layer];
    module = bit / module_size;
    bit    = bit % module_size;
  }
  else {
    module = -1;
  }
  
  if (target_type == TargetType::Weights) {
    const uint32_t element_size = topology.SIMD[layer] * topology.WPI[layer];
    const uint32_t element = bit / element_size;

    thresh = 0;
    ind    =  element % topology.WMEM[layer];
    mem    = (element / topology.WMEM[layer]) % topology.PE[layer];
    bit    = bit % element_size;
    layer  = layer * 2;
  }
  else {
    const uint32_t element_size = topology.activation_element_bits[layer];
    const uint32_t element = bit / element_size;

    thresh =   element % topology.API[layer];
    ind    =  (element / topology.API[layer]) % topology.TMEM[layer];
    mem    = ((element / topology.API[layer]) / topology.TMEM[layer]) % topology.PE[layer];
    bit    = bit % element_size;
    layer  = (layer * 2) + 1;
  }

  inject_fault_impl(layer, mem, ind, thresh, module, bit, word_size);
}


#if defined(OFFLOAD) && defined(RAWHLS)

#include "bnn-library.h"

void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, unsigned int doInit, unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps, unsigned int targetModule);

extern ExtMemWord * bufIn, * bufOut;

template<typename LowPrecType>
void FoldedMVOffload(const tiny_cnn::vec_t &in, tiny_cnn::vec_t & out, unsigned int offloadID, tiny_cnn::OffloadConvParams * convParams) {
  // binarize input and pack into bit stream
  binarizeAndPack(in, bufIn);

  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)bufIn, (ap_uint<64> *)bufOut, 0, 0, 0, 0, 0, 0, 1, -1);

  // unpack output bits and convert output back to float
  if(offloadID == 0xdeadbeef) {
    copyFromLowPrecBuffer<LowPrecType>((void *)bufOut, out);
  } else {
    unpackAndDebinarize(bufOut, out);
  }
}

template<unsigned int inWidth, unsigned int SIMDWidth, typename LowPrecType>
void FixedFoldedMVOffload(const tiny_cnn::vec_t &in, tiny_cnn::vec_t &out, unsigned int offloadID, tiny_cnn::OffloadConvParams * convParams) {
  // binarize input and pack into bit stream
  quantiseAndPack<inWidth, SIMDWidth>(in, bufIn);

  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)bufIn, (ap_uint<64> *)bufOut, 0, 0, 0, 0, 0, 0, 1, -1);

  // unpack output bits and convert output back to float
  if(offloadID == 0xdeadbeef) {
    copyFromLowPrecBuffer<LowPrecType>((void *)bufOut, out);
  } else {
    unpackAndDebinarize(bufOut, out);
  }
}


template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
void testPrebuiltCIFAR10(std::vector<tiny_cnn::vec_t> & imgs, std::vector<tiny_cnn::label_t> & labels, const unsigned int numCategories) {
  const unsigned int count = imgs.size();
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = 16; //paddedSize(numCategories*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi) {
    throw "Not enough space in accelBufIn";
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    throw "Not enough space in accelBufOut";
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  auto t1 = chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, 0, 0, 0, 0, 0,0, count, -1);
  auto t2 = chrono::high_resolution_clock::now();
  // compare against labels
  unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<LowPrecType>(&packedOut[i * pso], outTest);
    int maxInd = 0;
    LowPrecType maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
    if(maxInd == labels[i]) {
      ok++;
    } else {
      failed++;
    }
  }
  cout << "Succeeded " << ok << " failed " << failed << " accuracy " << 100.0*(float)ok/count << "%" << endl;
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  float usecPerImage = (float)duration / count;
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete[] packedImages;
  delete[] packedOut;
}


template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
std::vector<int>  testPrebuiltCIFAR10_from_image(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int numCategories, float &usecPerImage) {
  const unsigned int count = 1;
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi) {
    throw "Not enough space in accelBufIn";
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    throw "Not enough space in accelBufOut";
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;

  auto t1 = chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, 0, 0, 0, 0, 0, 0, count, -1);
  auto t2 = chrono::high_resolution_clock::now();

  // compare against labels
  unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
  copyFromLowPrecBuffer<LowPrecType>(&packedOut[0], outTest);
  std::vector<int> result;
  for(unsigned int j = 0; j < numCategories; j++) {
    result.push_back(outTest[j]);
  }
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete[] packedImages;
  delete[] packedOut;
  return result;
}

template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
std::vector<int> testPrebuiltCIFAR10_multiple_images(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int numCategories, std::vector<int> & detailed_results, float & usecPerImage) {
  const unsigned int count = imgs.size();
  std::vector<int> results;
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi)
    throw "Not enough space in accelBufIn";
  if(OUTPUT_BUF_ENTRIES < count*pso)
    throw "Not enough space in accelBufOut";
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  // copy inputs to accelerator
  auto t1 = chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, 0, 0, 0, 0, 0, 0, count, -1);
  auto t2 = chrono::high_resolution_clock::now();
  // compare against labels
  tiny_cnn::vec_t outTest(numCategories, 0);
  
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<LowPrecType>(&packedOut[i * pso], outTest);
    int maxInd = 0;
    LowPrecType maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
    detailed_results.push_back(outTest[j]);
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
	results.push_back(maxInd);
  }  
  auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete[] packedImages;
  delete[] packedOut;
  return results;
}

#elif defined(OFFLOAD) && !defined(RAWHLS)
#include "platform.hpp"
#include "interleave.h"
#include <vector>
#include <fstream>
#include <utility>

extern DonutDriver * thePlatform;
extern void * accelBufIn, * accelBufOut;
extern ExtMemWord * bufIn, * bufOut;

void ExecAccel();

template<size_t WeightSize>
void FoldedMVLoadInterleavedWeightPE(ifstream& wf,
  unsigned int layerNo, unsigned int pe, unsigned int linesWMem, unsigned int numModules,
  std::bitset<2*WeightSize> bit_pattern, const std::array<unsigned int, WeightSize>& e1_bit_order, const std::array<unsigned int, WeightSize>& e2_bit_order) {

  //const size_t half_weight_size = weight_size / 2;
  const uint64_t weight_bitmask = (uint64_t{1} << WeightSize) - 1;

  for(unsigned int line = 0 ; line < (linesWMem-1); line+=2) {
    ExtMemWord e1 = 0;
    ExtMemWord e2 = 0;
    wf.read((char *)&e1, sizeof(ExtMemWord));
    wf.read((char *)&e2, sizeof(ExtMemWord));

    /*
    const uint32_t e1_high = e1 >> half_weight_size;
    const uint32_t e1_low = e1 & ((uint64_t{1} << half_weight_size) - 1);
    const uint32_t e2_high = e2 >> half_weight_size;
    const uint32_t e2_low = e2 & ((uint64_t{1} << half_weight_size) - 1);

    const ExtMemWord high_interleaved = interleave(e1_high, e2_high);
    const ExtMemWord low_interleaved = interleave(e1_low, e2_low);
    */

    std::bitset<WeightSize> e1_bits(e1 & weight_bitmask);
    std::bitset<WeightSize> e2_bits(e2 & weight_bitmask);
    std::bitset<2*WeightSize> interleaved_bits = interleave_pattern(
      reorder(e1_bits, e1_bit_order),
      reorder(e2_bits, e2_bit_order),
      bit_pattern
    );

    const ExtMemWord e1_interleaved = (interleaved_bits >> WeightSize).to_ullong();
    const ExtMemWord e2_interleaved = interleaved_bits.to_ullong() & weight_bitmask;

    if (numModules > 0) { //write data for each module (used in triple-module redundancy version)
      for (unsigned int module = 0; module < numModules; module++) {
        FoldedMVMemSet(layerNo * 2, pe, line, 0, e1_interleaved, module);
        FoldedMVMemSet(layerNo * 2, pe, line+1, 0, e2_interleaved, module);
      }
    }
    else {
      FoldedMVMemSet(layerNo * 2, pe, line, 0, e1_interleaved);
      FoldedMVMemSet(layerNo * 2, pe, line+1, 0, e2_interleaved);
    }
  }

  // Write last element without interleaving if there's an odd number of elements
  if ((linesWMem % 2) != 0) {
    ExtMemWord e = 0;
    wf.read((char *)&e, sizeof(ExtMemWord));

    if (numModules > 0) { //write data for each module (used in triple-module redundancy version)
      for (unsigned int module = 0; module < numModules; module++) {
        FoldedMVMemSet(layerNo * 2, pe, linesWMem-1, 0, e, module);
      }
    }
    else {
      FoldedMVMemSet(layerNo * 2, pe, linesWMem-1, 0, e);
    }
  }
}

template<size_t ThreshSize>
void FoldedMVLoadInterleavedThreshPE(ifstream& tf,
  unsigned int layerNo, unsigned int pe, unsigned int linesTMem, unsigned int cntThresh, unsigned int numModules,
  std::bitset<2*ThreshSize> bit_pattern, const std::array<unsigned int, ThreshSize>& e1_bit_order, const std::array<unsigned int, ThreshSize>& e2_bit_order) {

  //const uint64_t half_thresh_size = ThreshSize / 2;
  const uint64_t thresh_bitmask = (uint64_t{1} << ThreshSize) - 1;

  std::vector<std::vector<ExtMemWord>> thresholds;

  // Load thresholds from file
  for(unsigned int line = 0 ; line < linesTMem; line++) {
    thresholds.emplace_back();

    for(unsigned int i = 0; i < cntThresh; ++i) {
      ExtMemWord e = 0;
      tf.read((char *)&e, sizeof(ExtMemWord));
      thresholds[line].push_back(e);
    }
  }

  // Interleave across linesTMem
  for(unsigned int line = 0 ; line < (linesTMem-1); line += 2) {
    for(unsigned int i = 0; i < cntThresh; ++i) {
      const ExtMemWord e1 = thresholds[line][i];
      const ExtMemWord e2 = thresholds[line+1][i];

      /*
      const uint32_t e1_high = e1 >> half_thresh_size;
      const uint32_t e1_low = e1 & ((1ull << half_thresh_size) - 1);
      const uint32_t e2_high = e2 >> half_thresh_size;
      const uint32_t e2_low = e2 & ((1ull << half_thresh_size) - 1);

      const ExtMemWord high_interleaved = interleave(e1_high, e2_high);
      const ExtMemWord low_interleaved = interleave(e1_low, e2_low);
      */

      std::bitset<ThreshSize> e1_bits(e1 & thresh_bitmask);
      std::bitset<ThreshSize> e2_bits(e2 & thresh_bitmask);
      std::bitset<2*ThreshSize> interleaved_bits = interleave_pattern(
        reorder(e1_bits, e1_bit_order),
        reorder(e2_bits, e2_bit_order),
        bit_pattern
      );

      const ExtMemWord e1_interleaved = (interleaved_bits >> ThreshSize).to_ullong();
      const ExtMemWord e2_interleaved = interleaved_bits.to_ullong() & thresh_bitmask;

      if (numModules > 0) {
        for (unsigned int module = 0; module < numModules; module++) {
          FoldedMVMemSet(layerNo * 2 + 1, pe, line, i, e1_interleaved, module);
          FoldedMVMemSet(layerNo * 2 + 1, pe, line+1, i, e2_interleaved, module);
        }
      }
      else {
        FoldedMVMemSet(layerNo * 2 + 1, pe, line, i, e1_interleaved);
        FoldedMVMemSet(layerNo * 2 + 1, pe, line+1, i, e2_interleaved);
      }
    }
  }

  // Write last array without interleaving if there's an odd number of arrays
  if ((linesTMem % 2) != 0) {
    for(unsigned int i = 0; i < cntThresh; ++i) {
      const ExtMemWord e = thresholds[linesTMem-1][i];
      if (numModules > 0) {
        for (unsigned int module = 0; module < numModules; module++) {
          FoldedMVMemSet(layerNo * 2 + 1, pe, linesTMem-1, i, e, module);
        }
      }
      else {
        FoldedMVMemSet(layerNo * 2 + 1, pe, linesTMem-1, i, e);
      }
    }
  }
}

template<typename T, T... I>
constexpr auto seq_to_array(std::integer_sequence<T, I...>) -> std::array<T, sizeof...(I)> {
  return std::array<T, sizeof...(I)>{ {I...} };
}
template<typename T, size_t N>
constexpr auto make_seq_array() -> std::array<T, N> {
  return seq_to_array(std::make_integer_sequence<T, N>{});
}

template<size_t WeightSize, size_t ThreshSize>
struct InterleavedLoadArgs {
  InterleavedLoadArgs(bool interleaved_weight, bool interleaved_thresh)
    : interleaved_weight(interleaved_weight)
    , interleaved_thresh(interleaved_thresh) {
  }

  InterleavedLoadArgs(bool interleaved_weight,
                      bool interleaved_thresh,
                      std::bitset<2*WeightSize> weight_pattern,
                      std::bitset<2*ThreshSize> thresh_pattern,
                      const std::array<unsigned int, WeightSize>& e1_weight_order,
                      const std::array<unsigned int, WeightSize>& e2_weight_order,
                      const std::array<unsigned int, ThreshSize>& e1_thresh_order,
                      const std::array<unsigned int, ThreshSize>& e2_thresh_order)
    : interleaved_weight(interleaved_weight)
    , interleaved_thresh(interleaved_thresh)
    , weight_pattern(weight_pattern)
    , thresh_pattern(thresh_pattern)
    , e1_weight_order(e1_weight_order)
    , e2_weight_order(e2_weight_order)
    , e1_thresh_order(e1_thresh_order)
    , e2_thresh_order(e2_thresh_order) {
  }

  bool interleaved_weight;
  bool interleaved_thresh;
  std::bitset<2*WeightSize> weight_pattern = 0x5555555555555555;
  std::bitset<2*ThreshSize> thresh_pattern = 0x5555555555555555;
  std::array<unsigned int, WeightSize> e1_weight_order = make_seq_array<unsigned int, WeightSize>();
  std::array<unsigned int, WeightSize> e2_weight_order = make_seq_array<unsigned int, WeightSize>();
  std::array<unsigned int, ThreshSize> e1_thresh_order = make_seq_array<unsigned int, ThreshSize>();
  std::array<unsigned int, ThreshSize> e2_thresh_order = make_seq_array<unsigned int, ThreshSize>();
};

template<size_t WeightSize, size_t ThreshSize>
void FoldedMVLoadInterleavedLayerMem(std::string dir, 
                                     unsigned int layerNo,
                                     unsigned int peCount, 
                                     unsigned int linesWMem, 
                                     unsigned int linesTMem, 
                                     unsigned int numThresh,
                                     unsigned int numModules,
                                     const InterleavedLoadArgs<WeightSize, ThreshSize>& interleave_args) {
  for(unsigned int pe = 0; pe < peCount; pe++) {
    // load weights
    ifstream wf(dir + "/" + to_string(layerNo) + "-" + to_string(pe) + "-weights.bin", ios::binary | ios::in);
    if(!wf.is_open()) {
      throw "Could not open file";
    }
    if (interleave_args.interleaved_weight) {
      FoldedMVLoadInterleavedWeightPE<WeightSize>(wf, layerNo, pe, linesWMem, numModules, interleave_args.weight_pattern, interleave_args.e1_weight_order, interleave_args.e2_weight_order);
    }
    else {
      FoldedMVLoadWeightPE(wf, layerNo, pe, linesWMem, numModules);
    }
    wf.close();

    // load thresholds
    if(numThresh > 0) {
      ifstream tf(dir + "/" + to_string(layerNo) + "-" + to_string(pe) + "-thres.bin", ios::binary | ios::in);
      if(!tf.is_open()) {
        throw "Could not open file";
      }
      if (interleave_args.interleaved_thresh) {
        FoldedMVLoadInterleavedThreshPE<ThreshSize>(tf, layerNo, pe, linesTMem, numThresh, numModules, interleave_args.thresh_pattern, interleave_args.e1_thresh_order, interleave_args.e2_thresh_order);
      }
      else {
        FoldedMVLoadThreshPE(tf, layerNo, pe, linesTMem, numThresh, numModules);
      }
      tf.close();
    }
  }
}

template<typename LowPrecType>
void FoldedMVOffload(const tiny_cnn::vec_t &in, tiny_cnn::vec_t &out, unsigned int offloadID, tiny_cnn::OffloadConvParams * convParams) {
  // always operates on a single image per call for now -- set numImages to 1
  thePlatform->writeJamRegAddr(0x5C, 1);
  // binarize input and pack into bit stream
  binarizeAndPack(in, bufIn);

  // TODO size to pad input to is max(64, PE_SYNGROUP_BITS)
  unsigned int paddedInDim = paddedSize(in.size(), bitsPerExtMemWord);
  // copy into accelerator input
  const unsigned int numInpWords = (paddedInDim / bitsPerExtMemWord);
  thePlatform->copyBufferHostToAccel((void *)bufIn, accelBufIn, sizeof(ExtMemWord) * numInpWords);

  // launch
  ExecAccel();

  if(offloadID == 0xdeadbeef) {
    unsigned int paddedOutDim = paddedSize(out.size() * 16, bitsPerExtMemWord);
    const unsigned int numOutWords = (paddedOutDim / bitsPerExtMemWord);
    thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord) * numOutWords);
    copyFromLowPrecBuffer<LowPrecType>((void *)bufOut, out);
  } else {
    // TODO size to pad input to is max(64, NUM_PE_ELEMENTS)
    unsigned int paddedOutDim = paddedSize(out.size(), bitsPerExtMemWord);

    // copy from accelerator output
    const unsigned int numOutWords = ( paddedOutDim / bitsPerExtMemWord);
    thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord) * numOutWords);

    // unpack output bits and convert output back to float
    unpackAndDebinarize(bufOut, out);
  }
}

template<unsigned int inWidth, unsigned int SIMDWidth, typename LowPrecType>
void FixedFoldedMVOffload(const tiny_cnn::vec_t &in, tiny_cnn::vec_t &out, unsigned int offloadID, tiny_cnn::OffloadConvParams * convParams) {
  // always operates on a single image per call for now -- set numImages to 1
  thePlatform->writeJamRegAddr(0x5C, 1);
  // binarize input and pack into bit stream
  quantiseAndPack<inWidth, SIMDWidth>(in, bufIn);

  // TODO size to pad input to is max(64, PE_SYNGROUP_BITS)
  unsigned int paddedInDim = paddedSize(in.size(), bitsPerExtMemWord);
  // copy into accelerator input
  const unsigned int numInpWords = (paddedInDim / (bitsPerExtMemWord / inWidth));
  thePlatform->copyBufferHostToAccel((void *)bufIn, accelBufIn, sizeof(ExtMemWord) * numInpWords);

  // launch
  ExecAccel();

  if(offloadID == 0xdeadbeef) {
    unsigned int paddedOutDim = paddedSize(out.size() * 16, bitsPerExtMemWord);
    const unsigned int numOutWords = ( paddedOutDim / bitsPerExtMemWord);
    thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord) * numOutWords);
    copyFromLowPrecBuffer<LowPrecType>((void *)bufOut, out);
  } else {
    // TODO size to pad input to is max(64, NUM_PE_ELEMENTS)
    unsigned int paddedOutDim = paddedSize(out.size(), bitsPerExtMemWord);

    // copy from accelerator output
    const unsigned int numOutWords = ( paddedOutDim / bitsPerExtMemWord);
    thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord) * numOutWords);

    // unpack output bits and convert output back to float
    unpackAndDebinarize(bufOut, out);
  }
}

template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
void testPrebuiltCIFAR10(std::vector<tiny_cnn::vec_t> & imgs, std::vector<tiny_cnn::label_t> & labels, const unsigned int numCategories) {
  const unsigned int count = imgs.size();
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // # of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size() * inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // # of ExtMemWords per output
  const unsigned int pso = paddedSize(numCategories * outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi) {
    throw "Not enough space in accelBufIn";
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    throw "Not enough space in accelBufOut";
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  // copy inputs to accelerator
  thePlatform->copyBufferHostToAccel((void *)packedImages, accelBufIn, sizeof(ExtMemWord) * count * psi);
  // set number of images to recognize
  thePlatform->writeJamRegAddr(0x5C, count);
  
  // recognize
  auto t1 = chrono::high_resolution_clock::now();
  ExecAccel();
  auto t2 = chrono::high_resolution_clock::now();
  
  // copy results back to host
  thePlatform->copyBufferAccelToHost(accelBufOut, (void *)packedOut, sizeof(ExtMemWord) * count * pso);
  // compare against labels
  unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<LowPrecType>(&packedOut[i * pso], outTest);
    int maxInd = 0;
    LowPrecType maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
    if(maxInd == labels[i]) {
      ok++;
    } else {
      failed++;
    }
  }
  cout << "Succeeded " << ok << " failed " << failed << " accuracy " << 100.0 * (float)ok / count << "%" << endl;
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  float usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete[] packedImages;
  delete[] packedOut;
}

template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
std::vector<int> testPrebuiltCIFAR10_from_image(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int numCategories, float &usecPerImage) {
  const unsigned int count = 1;
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi) {
    throw "Not enough space in accelBufIn";
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    throw "Not enough space in accelBufOut";
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  // copy inputs to accelerator
  thePlatform->copyBufferHostToAccel((void *)packedImages, accelBufIn, sizeof(ExtMemWord) * count * psi);
  // set number of images to recognize
  thePlatform->writeJamRegAddr(0x5C, count);
  
  // recognize
  auto t1 = chrono::high_resolution_clock::now();
  ExecAccel();
  auto t2 = chrono::high_resolution_clock::now();
  
  // copy results back to host
  thePlatform->copyBufferAccelToHost(accelBufOut, (void *)packedOut, sizeof(ExtMemWord) * count * pso);

  // compare against labels
  unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
  copyFromLowPrecBuffer<LowPrecType>(&packedOut[0], outTest);
  std::vector<int> result;
  for(unsigned int j = 0; j < numCategories; j++) {
    result.push_back(outTest[j]);
  }

  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete [] packedImages;
  delete [] packedOut;
  return result;
}

template<unsigned int inWidth, unsigned int outWidth, typename LowPrecType>
std::vector<int> testPrebuiltCIFAR10_multiple_images(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int numCategories, std::vector<int> & detailed_results, float &usecPerImage) {
  const unsigned int count = imgs.size();
  std::vector<int> results;
  cout << "Packing and interleaving CIFAR-""10 inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi) {
    throw "Not enough space in accelBufIn";
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    throw "Not enough space in accelBufOut";
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  // copy inputs to accelerator
  thePlatform->copyBufferHostToAccel((void *)packedImages, accelBufIn, sizeof(ExtMemWord) * count * psi);
  // set number of images to recognize
  thePlatform->writeJamRegAddr(0x5C, count);
  
  // recognize
  auto t1 = chrono::high_resolution_clock::now();
  ExecAccel();
  auto t2 = chrono::high_resolution_clock::now();
  
  // copy results back to host
  thePlatform->copyBufferAccelToHost(accelBufOut, (void *)packedOut, sizeof(ExtMemWord) * count * pso);
  tiny_cnn::vec_t outTest(numCategories, 0);
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<LowPrecType>(&packedOut[i * pso], outTest);
    int maxInd = 0;
    LowPrecType maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
    detailed_results.push_back(outTest[j]);
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
    results.push_back(maxInd);	   	  
  }  

  auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete[] packedImages;
  delete[] packedOut;
  return results;
 }


#endif

