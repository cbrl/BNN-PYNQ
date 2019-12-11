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

struct RandomFaultArgs {
	RandomFaultArgs(
		const NetworkTopology& topology,
		std::vector<uint32_t> target_layers,
		TargetType target_type,
		bool flip_word,
		std::function<void(const NetworkTopology&, TargetType, bool, uint32_t, uint32_t)> injection_func
	)
	: topology(topology)
	, target_layers(target_layers)
	, target_type(target_type)
	, flip_word(flip_word)
	, injection_func(injection_func) {
	}

	const NetworkTopology& topology;
	std::vector<uint32_t> target_layers;
	TargetType target_type;
	bool flip_word;
	std::function<void(const NetworkTopology&, TargetType, bool, uint32_t, uint32_t)> injection_func;
};

namespace fault_impl {
void inject_random_fault(const RandomFaultArgs& args) {
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

	return args.injection_func(args.topology, target, args.flip_word, layer, bit);
}
}

template<typename ClassificationFuncT>
std::function<void(uint32_t, uint32_t)> make_faulty_classification_func(
	const RandomFaultArgs& fault_args,
	ClassificationFuncT&& classification_func
) {
	return [=](uint32_t num_faults, uint32_t num_classifications) {
		// Random device/generator
		std::random_device rd;
		std::mt19937 gen{rd()};

		// Used to determine what time in the process to inject a fault
		std::uniform_int_distribution<size_t> loop_dist{0, num_classifications - 1};

		// Generate fault times
		std::vector<size_t> fault_indices;
		for (unsigned int i = 0; i < num_faults; ++i) {
			fault_indices.push_back(loop_dist(gen));
		}

		for (uint32_t i = 0; i < num_classifications; ++i) {
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