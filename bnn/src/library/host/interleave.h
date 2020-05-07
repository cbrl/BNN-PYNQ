#pragma once

#include <array>
#include <bitset>
#include <type_traits>

// Interleaves the bits of two values according to a pattern. If a bit in the
// pattern is 1, then the corresponding bit in the result will be the next bit
// in x. If the bit is 0, then it will be the next bit from y.
template<size_t N>
std::bitset<2*N> interleave_pattern(const std::bitset<N>& x, const std::bitset<N>& y, const std::bitset<2*N>& pattern) {
	std::bitset<2*N> out;
	size_t x_idx = 0;
	size_t y_idx = 0;

	for (size_t i = 0; i < pattern.size(); ++i) {
		out[i] = pattern[i] ? x[x_idx++] : y[y_idx++];
	}

	return out;
}


// Interleaves the bits of two values. The bits of x will be in the even
// positions of the resultant value (0, 2, 4, ...), and the bits of y
// will be in the odd positions.
template<size_t N>
std::enable_if_t<(N == 1) || ((N & (N-1)) != 0), std::bitset<2*N>> //Bit size not a power of 2
interleave(const std::bitset<N>& x, const std::bitset<N>& y) {
	std::bitset<2*N> out;
	for (size_t i = 0; i < N; ++i) {
		out[2*i] = x[i];
		out[(2*i)+1] = y[i];
	}
	return out;
}

// Interleaves the bits of two values. The bits of x will be in the even
// positions of the resultant value (0, 2, 4, ...), and the bits of y
// will be in the odd positions.
template<size_t N>
std::enable_if_t<(N > 1) && ((N & (N-1)) == 0), std::bitset<2*N>> //Bit size is a power of 2
interleave(const std::bitset<N>& x_in, const std::bitset<N>& y_in) {
    std::bitset<2*N> x = x_in.to_ullong();
    std::bitset<2*N> y = y_in.to_ullong();
    std::bitset<2*N> mask;
	mask.flip();

	size_t s = N; //bit size must be power of 2

	do {
		mask ^= (mask << s);
		x = (x | (x << s)) & mask;
		y = (y | (y << s)) & mask;
	} while ((s >>= 1) > 0);

    return x | (y << 1);
}



template<size_t N>
std::enable_if_t<(N == 1) || ((N & (N-1)) != 0), std::bitset<N>> //Bit size is not a power of 2
reverse(std::bitset<N> bits) {
	bool temp;
	for (size_t i = 0; i < N/2; ++i) {
		temp = bits[i];
		bits[i] = bits[N-i-1];
		bits[N-i-1] = temp;
	}

	return bits;
}


template<size_t N>
std::enable_if_t<(N > 1) && ((N & (N-1)) == 0), std::bitset<N>> //Bit size is a power of 2
reverse(std::bitset<N> bits) {
	size_t s = N; //bit size must be power of 2
	std::bitset<N> mask;
	mask.flip();

	while ((s >>= 1) > 0) {
		mask ^= (mask << s);
		bits = ((bits >> s) & mask) | ((bits << s) & ~mask);
	}

	return bits;
}



// Reorders the bits in a value according to a user provided order. Index
// 0 of the order array corresponds to the least significant bit value to reorder,
// so {8, 7, 6, ..., 1, 0} would reverse the bits of an 8-bit value.
template<size_t N>
std::bitset<N> reorder(std::bitset<N> bits, const std::array<unsigned int, N>& order) {
	const std::bitset<N> original = bits;

	for (size_t i = 0; i < N; ++i) {
		bits[i] = original[order[i]];
	}

	return bits;
}