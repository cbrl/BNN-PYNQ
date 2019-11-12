#pragma once

#include <random>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

struct layer_data {
	layer_data(
		std::initializer_list<uint32_t> weight_modules,
		std::initializer_list<uint32_t> activation_modules,
		std::initializer_list<uint32_t> PE,
		std::initializer_list<uint32_t> WMEM,
		std::initializer_list<uint32_t> TMEM,
		std::initializer_list<uint32_t> SIMD,
		std::initializer_list<uint32_t> API,
		std::initializer_list<uint32_t> WPI,
		std::initializer_list<uint32_t> activation_element_bits
	)
	: weight_modules(weight_modules)
	, activation_modules(activation_modules)
	, PE(PE)
	, WMEM(WMEM)
	, TMEM(TMEM)
	, SIMD(SIMD)
	, API(API)
	, WPI(WPI)
	, activation_element_bits(activation_element_bits) {
		for (size_t i = 0; i < WPI.size(); ++i) {
			weight_bit_sizes.push_back(this->weight_modules[i] * this->WPI[i] * this->SIMD[i] * this->PE[i] * this->WMEM[i]);
		}
		for (size_t i = 0; i < activation_element_bits.size(); ++i) {
			activation_bit_sizes.push_back(this->activation_modules[i] * this->TMEM[i] * this->PE[i] * this->API[i] * this->activation_element_bits[i]);
		}
	}

	uint32_t weight_bits() const noexcept {
		static const uint32_t bits = std::accumulate(weight_bit_sizes.begin(), weight_bit_sizes.end(), 0);
		return bits;
	}

	uint32_t weight_bits(const std::vector<uint32_t>& layers) const {
		uint32_t bits = 0;
		for (const auto& layer : layers) {
			bits += weight_bit_sizes[layer];
		}
		return bits;
	}

	uint32_t activation_bits() const noexcept {
		static const uint32_t bits = std::accumulate(activation_bit_sizes.begin(), activation_bit_sizes.end(), 0);
		return bits;
	}

	uint32_t activation_bits(const std::vector<uint32_t>& layers) const {
		uint32_t bits = 0;
		for (const auto& layer : layers) {
			bits += activation_bit_sizes[layer];
		}
		return bits;
	}

	std::vector<uint32_t> weight_modules;
	std::vector<uint32_t> activation_modules;
	std::vector<uint32_t> PE;
	std::vector<uint32_t> WMEM;
	std::vector<uint32_t> TMEM;
	std::vector<uint32_t> SIMD;
	std::vector<uint32_t> API;
	std::vector<uint32_t> WPI;
	std::vector<uint32_t> activation_element_bits;
	std::vector<uint32_t> weight_bit_sizes;
	std::vector<uint32_t> activation_bit_sizes;
};



uint32_t random_weighted_selection(const std::vector<uint32_t>& weights, std::mt19937& gen) {
	std::discrete_distribution<uint32_t> weighted_dist{
		std::begin(weights), std::end(weights)
	};
	return weighted_dist(gen);
}

uint32_t random_weighted_selection(const std::vector<uint32_t>& weights, const std::vector<uint32_t>& indices, std::mt19937& gen) {
	std::vector<uint32_t> filtered_weights;
	for (const auto& index : indices) {
		filtered_weights.push_back(weights[index]);
	}

	std::discrete_distribution<uint32_t> weighted_dist{std::begin(filtered_weights), std::end(filtered_weights)};
	return indices[weighted_dist(gen)];
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>
random_weight(const layer_data& layers, uint32_t layer, std::mt19937& gen) {
	std::uniform_int_distribution<uint32_t> pe_dist{0, layers.PE[layer] - 1};
	std::uniform_int_distribution<uint32_t> wmem_dist{0, layers.WMEM[layer] - 1};
	std::uniform_int_distribution<uint32_t> bit_dist{0, (layers.SIMD[layer] * layers.WPI[layer]) - 1};

	return {pe_dist(gen), wmem_dist(gen), 0, bit_dist(gen)};
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>
random_activation(const layer_data& layers, uint32_t layer, std::mt19937& gen) {
	std::uniform_int_distribution<uint32_t> pe_dist{0, layers.PE[layer] - 1};
	std::uniform_int_distribution<uint32_t> tmem_dist{0, layers.TMEM[layer] - 1};
	std::uniform_int_distribution<uint32_t> api_dist{0, layers.API[layer] - 1};
	std::uniform_int_distribution<uint32_t> bit_dist{0, layers.activation_element_bits[layer] - 1};

	return {pe_dist(gen), tmem_dist(gen), api_dist(gen), bit_dist(gen)};
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
random_selection(const layer_data& layers, int target_type) {
	std::random_device rd;
	std::mt19937 gen{rd()};

	bool weight;
	if (target_type < 0) {
		std::discrete_distribution<uint32_t> weight_or_activation{
			{static_cast<double>(layers.weight_bits()), static_cast<double>(layers.activation_bits())}
		};
		weight = weight_or_activation(gen) == 0;
	}
	else {
		weight = (target_type == 0);
	}

	if (weight) {
		const auto layer = random_weighted_selection(layers.weight_bit_sizes, gen);
		const auto weight = random_weight(layers, layer, gen);
		return std::tuple_cat(std::make_tuple(2*layer), weight);
	}
	else {
		const auto layer = random_weighted_selection(layers.activation_bit_sizes, gen);
		const auto activation = random_activation(layers, layer, gen);
		return std::tuple_cat(std::make_tuple((2*layer)+1), activation);
	}
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
random_selection(const layer_data& layers, int target_type, const std::vector<uint32_t>& target_layers) {
	std::random_device rd;
	std::mt19937 gen{rd()};

	const uint32_t weight_space = layers.weight_bits(target_layers);
	const uint32_t activation_space = layers.activation_bits(target_layers);

	bool weight;
	if (target_type < 0) {
		std::discrete_distribution<uint32_t> weight_or_activation{
			{static_cast<double>(weight_space), static_cast<double>(activation_space)}
		};
		weight = weight_or_activation(gen) == 0;
	}
	else {
		weight = (target_type == 0);
	}

	if (weight) {
		const auto layer = random_weighted_selection(layers.weight_bit_sizes, target_layers, gen);
		const auto weight = random_weight(layers, layer, gen);
		return std::tuple_cat(std::make_tuple(2*layer), weight);
	}
	else {
		const auto layer = random_weighted_selection(layers.activation_bit_sizes, target_layers, gen);
		const auto activation = random_activation(layers, layer, gen);
		return std::tuple_cat(std::make_tuple((2*layer)+1), activation);
	}
}