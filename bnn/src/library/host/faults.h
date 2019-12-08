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

template<typename InjectionReturnT>
InjectionReturnT inject_random_fault(
	const NetworkTopology& topology,
	const std::vector<uint32_t>& target_layers,
	TargetType target_type,
	bool flip_word,
	const std::function<InjectionReturnT(const NetworkTopology&, TargetType, bool, uint32_t, uint32_t)>& injection_func
) {
	if (target_type == TargetType::Any) {
		std::random_device rd;
		std::mt19937 gen{rd()};

		const uint32_t weight_space = topology.weight_bits(target_layers);
		const uint32_t activation_space = topology.activation_bits(target_layers);

		std::discrete_distribution<uint32_t> weight_or_activation{
			{static_cast<double>(weight_space), static_cast<double>(activation_space)}
		};

		target_type = (weight_or_activation(gen) == 0) ? TargetType::Weights : TargetType::Activations;
	}

    uint32_t layer;
    uint32_t bit;
	if (target_type == TargetType::Weights) {
		const auto selection = topology.random_weight_bit(target_layers);
        layer = std::get<0>(selection);
        bit = std::get<1>(selection);
	}
	else if (target_type == TargetType::Activations) {
		const auto selection = topology.random_activation_bit(target_layers);
        layer = std::get<0>(selection);
        bit = std::get<1>(selection);
	}

	return injection_func(topology, target_type, flip_word, layer, bit);
}