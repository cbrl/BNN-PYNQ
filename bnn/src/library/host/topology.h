#pragma once

#include <random>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>
#include <array>


class NetworkTopology {
public:
	// Returns <layer, bit_in_layer>
	virtual std::tuple<uint32_t, uint32_t> random_weight_bit() const noexcept = 0;
	virtual std::tuple<uint32_t, uint32_t> random_weight_bit(const std::vector<uint32_t>& target_layers) const noexcept = 0;

	// Returns <layer, bit_in_layer>
	virtual std::tuple<uint32_t, uint32_t> random_activation_bit() const noexcept = 0;
	virtual std::tuple<uint32_t, uint32_t> random_activation_bit(const std::vector<uint32_t>& target_layers) const noexcept = 0;

	virtual uint32_t weight_bits() const noexcept = 0;
	virtual uint32_t weight_bits(const std::vector<uint32_t>& layers) const noexcept = 0;

	virtual uint32_t activation_bits() const noexcept = 0;
	virtual uint32_t activation_bits(const std::vector<uint32_t>& layers) const noexcept = 0;
};


template<size_t NumLayers>
class FINNTopology final : public NetworkTopology {
public:
	static constexpr size_t num_layers = NumLayers;

	FINNTopology(
		std::array<uint32_t, NumLayers> num_weight_modules,
		std::array<uint32_t, NumLayers> num_activation_modules,
		std::array<uint32_t, NumLayers> PE,
		std::array<uint32_t, NumLayers> WMEM,
		std::array<uint32_t, NumLayers> TMEM,
		std::array<uint32_t, NumLayers> SIMD,
		std::array<uint32_t, NumLayers> API,
		std::array<uint32_t, NumLayers> WPI,
		std::array<uint32_t, NumLayers> activation_element_bits
	)
	: weight_modules(std::move(num_weight_modules))
	, activation_modules(std::move(num_activation_modules))
	, PE(std::move(PE))
	, WMEM(std::move(WMEM))
	, TMEM(std::move(TMEM))
	, SIMD(std::move(SIMD))
	, API(std::move(API))
	, WPI(std::move(WPI))
	, activation_element_bits(std::move(activation_element_bits))
	, gen(std::random_device{}()) {
		for (size_t i = 0; i < num_layers; ++i) {
			weight_bit_sizes[i] = this->weight_modules[i] * this->WPI[i] * this->SIMD[i] * this->PE[i] * this->WMEM[i];
		}
		for (size_t i = 0; i < num_layers; ++i) {
			activation_bit_sizes[i] = this->activation_modules[i] * this->TMEM[i] * this->PE[i] * this->API[i] * this->activation_element_bits[i];
		}
	}

	virtual uint32_t weight_bits() const noexcept override {
		static const uint32_t bits = std::accumulate(weight_bit_sizes.begin(), weight_bit_sizes.end(), 0);
		return bits;
	}

	virtual uint32_t weight_bits(const std::vector<uint32_t>& layers) const noexcept override {
		uint32_t bits = 0;

		const size_t index_max = std::min(layers.size(), weight_bit_sizes.size());
		for (size_t i = 0; i < index_max; ++i) {
			bits += weight_bit_sizes[layers[i]];
		}

		return bits;
	}

	virtual uint32_t activation_bits() const noexcept override {
		static const uint32_t bits = std::accumulate(activation_bit_sizes.begin(), activation_bit_sizes.end(), 0);
		return bits;
	}

	virtual uint32_t activation_bits(const std::vector<uint32_t>& layers) const noexcept override {
		uint32_t bits = 0;

		const size_t index_max = std::min(layers.size(), activation_bit_sizes.size());
		for (size_t i = 0; i < index_max; ++i) {
			bits += activation_bit_sizes[layers[i]];
		}

		return bits;
	}

	virtual std::tuple<uint32_t, uint32_t> random_weight_bit() const noexcept override {
		const uint32_t layer = random_weighted_selection(weight_bit_sizes);
		const uint32_t bit = random_bit(weight_bit_sizes[layer]);
		return {layer, bit};
	}

	virtual std::tuple<uint32_t, uint32_t> random_weight_bit(const std::vector<uint32_t>& target_layers) const noexcept override {
		const uint32_t layer = random_weighted_selection(weight_bit_sizes, target_layers);
		const uint32_t bit = random_bit(weight_bit_sizes[layer]);
		return {layer, bit};
	}

	virtual std::tuple<uint32_t, uint32_t> random_activation_bit() const noexcept override {
		const uint32_t layer = random_weighted_selection(activation_bit_sizes);
		const uint32_t bit = random_bit(activation_bit_sizes[layer]);
		return {layer, bit};
	}

	virtual std::tuple<uint32_t, uint32_t> random_activation_bit(const std::vector<uint32_t>& target_layers) const noexcept override {
		const uint32_t layer = random_weighted_selection(activation_bit_sizes, target_layers);
		const uint32_t bit = random_bit(activation_bit_sizes[layer]);
		return {layer, bit};
	}

private:

	uint32_t random_weighted_selection(const std::array<uint32_t, NumLayers>& weights) const {
		std::discrete_distribution<uint32_t> weighted_dist{
			std::begin(weights), std::end(weights)
		};
		return weighted_dist(gen);
	}

	uint32_t random_weighted_selection(const std::array<uint32_t, NumLayers>& weights, const std::vector<uint32_t>& indices) const {
		if (indices.size() == 0) {
			return random_weighted_selection(weights);
		}

		std::vector<uint32_t> filtered_weights;
		for (const auto& index : indices) {
			filtered_weights.push_back(weights[index]);
		}

		std::discrete_distribution<uint32_t> weighted_dist{std::begin(filtered_weights), std::end(filtered_weights)};
		return indices[weighted_dist(gen)];
	}

	uint32_t random_bit(uint32_t size) const noexcept {
		std::uniform_int_distribution<uint32_t> bit_dist{0, size - 1};
		return bit_dist(gen);
	}

public:

	std::array<uint32_t, NumLayers> weight_modules;
	std::array<uint32_t, NumLayers> activation_modules;
	std::array<uint32_t, NumLayers> PE;
	std::array<uint32_t, NumLayers> WMEM;
	std::array<uint32_t, NumLayers> TMEM;
	std::array<uint32_t, NumLayers> SIMD;
	std::array<uint32_t, NumLayers> API;
	std::array<uint32_t, NumLayers> WPI;
	std::array<uint32_t, NumLayers> activation_element_bits;
	std::array<uint32_t, NumLayers> weight_bit_sizes;
	std::array<uint32_t, NumLayers> activation_bit_sizes;

private:

	mutable std::mt19937 gen;
};

using CNVTopology = FINNTopology<9>;
using LFCTopology = FINNTopology<4>;