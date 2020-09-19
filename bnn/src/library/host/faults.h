#pragma once

#include <functional>
#include <random>
#include <vector>
#include "topology.h"


enum class TargetType : uint8_t {
	Any = 0,
	Weights = 1,
	Activations = 2,
};


// RandomFaultArgs
//
// Description:
// A struct which contains the arguments needed for fault injection
//
// Members:
// topology       - The filled NetworkTopology class
//
// target_layers  - A vector containing the layers that should
//                  be targeted. Pass empty vector for all layers.
//
// target_type    - Indicates if weights, activations, or both should
//                  be targeted for fault injection.
//
// word_size      - The size of the word to flip, in bits. (Set to 1 to flip
//                  a bit instead of a word).
//
// injection_func - A function which handles the actual fault injection.
//                  arguments are (topology, target_type, word_size, layer, bit).
//                  Layer and bit are the n-th layer and n-th bit in that layer.
struct RandomFaultArgs {
	RandomFaultArgs(
		const NetworkTopology& topology,
		std::vector<uint32_t> target_layers,
		TargetType target_type,
		uint8_t word_size,
		std::function<void(const NetworkTopology&, TargetType, bool, uint32_t, uint32_t)> injection_func
	)
	: topology(topology)
	, target_layers(target_layers)
	, target_type(target_type)
	, word_size(word_size)
	, injection_func(injection_func) {
	}

	const NetworkTopology& topology;
	std::vector<uint32_t> target_layers;
	TargetType target_type;
	uint8_t word_size;
	std::function<void(const NetworkTopology&, TargetType, uint8_t, uint32_t, uint32_t)> injection_func;
};


namespace fault_impl {
inline void inject_random_fault(const RandomFaultArgs& args) {
	TargetType target;

	if (args.target_type == TargetType::Any) {
		std::random_device rd;
		std::mt19937 gen{rd()};

		const uint32_t weight_space = args.topology.weight_bits(args.target_layers);
		const uint32_t activation_space = args.topology.activation_bits(args.target_layers);

		std::discrete_distribution<uint32_t> weight_or_activation{
			{static_cast<double>(weight_space), static_cast<double>(activation_space)}
		};

		target = (weight_or_activation(gen) == 0) ? TargetType::Weights : TargetType::Activations;
	}
	else {
		target = args.target_type;
	}

    uint32_t layer;
    uint32_t bit;
	if (target == TargetType::Weights) {
		const auto selection = args.topology.random_weight_bit(args.target_layers);
        layer = std::get<0>(selection);
        bit = std::get<1>(selection);
	}
	else if (target == TargetType::Activations) {
		const auto selection = args.topology.random_activation_bit(args.target_layers);
        layer = std::get<0>(selection);
        bit = std::get<1>(selection);
	}

	return args.injection_func(args.topology, target, args.word_size, layer, bit);
}
}


// make_faulty_classification_func
//
// Description:
// Create a function which classifies images and injects faults at uniformly
// distributed locations and times.
//
// Arguments:
// fault_args          - A filled RandomFaultArgs struct
// num_images          - The number of images to classify
// classification_func - A function which accepts an index as an argument
//                       and classifies the associated image
//
// Returns:
// A std::function which runs the classifcation and automatically injects
// faults. It accepts one argument, which specified the number of faults
// to inject. The faults will be uniformly distributed based on the
// number of images to classify.
template<typename ClassificationFuncT>
std::function<void(uint32_t)> make_faulty_classification_func(
	const RandomFaultArgs& fault_args,
	size_t num_images,
	ClassificationFuncT&& classification_func
) {
	return [=](uint32_t num_faults) {
		// Random device/generator
		std::random_device rd;
		std::mt19937 gen{rd()};

		// Used to determine what time in the process to inject a fault
		std::uniform_int_distribution<size_t> loop_dist{0, num_images - 1};

		// Generate fault times
		std::vector<size_t> fault_indices;
		for (unsigned int i = 0; i < num_faults; ++i) {
			fault_indices.push_back(loop_dist(gen));
		}

		for (uint32_t i = 0; i < num_images; ++i) {
			// Inject a fault
			std::vector<size_t>::iterator it = std::find(fault_indices.begin(), fault_indices.end(), i);
			while (it != fault_indices.end()) {
				fault_impl::inject_random_fault(fault_args);

				fault_indices.erase(it);
				it = std::find(fault_indices.begin(), fault_indices.end(), i);
			}

			classification_func(i);
		}
	};
}