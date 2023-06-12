/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2023 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Input features and network structure used in NNUE evaluation function

#ifndef NNUE_ARCHITECTURE_H_INCLUDED
#define NNUE_ARCHITECTURE_H_INCLUDED

#include <memory>
#include <chrono>
#include <atomic>

#include "nnue_common.h"
#include "features/half_ka_v2_hm.h"

#include "layers/affine_transform.h"
#include "layers/clipped_relu.h"
#include "layers/sqr_clipped_relu.h"

#include "../misc.h"

extern std::atomic<uint64_t> total_nnue_ns;
extern std::atomic<uint64_t> total_nnue_invocations; 

namespace Stockfish::Eval::NNUE {

// Input features used in evaluation function
using FeatureSet = Features::HalfKAv2_hm;

// Number of input feature dimensions after conversion
constexpr IndexType TransformedFeatureDimensions = 1024;
constexpr IndexType PSQTBuckets = 8;
constexpr IndexType LayerStacks = 8;


#ifdef _MSC_VER
#define restrict __restrict
#else 
#define restrict __restrict__
#endif


static inline int propagate_scalar(const uint8_t* input, int8_t* w0, int32_t* b0, int8_t* w1, int32_t* b1, int8_t* w2, int32_t* b2)
{
	constexpr int Layer0_PaddedInputDimensions = 1024;
	constexpr int Layer0_InputDimensions = 1024;
	constexpr int Layer1_PaddedInputDimensions = 32;
	constexpr int Layer1_InputDimensions = 30;
	constexpr int weight_scale = 6;

	uint8_t buffer[30]; //32 for simd version
	uint8_t* input0 = buffer;
	uint8_t* input1 = buffer + 15;
	for (int i = 0; i < 15; ++i) {
		int8_t* pos_ptr = w0 + i * 1024;
		std::int32_t sum = b0[i];
		for (int j = 0; j < Layer0_InputDimensions; ++j) {
			sum += pos_ptr[j] * input[j];
		}
		input0[i] = static_cast<uint8_t>(std::max(0ll, std::min(127ll, (((long long)sum * sum) >> (2 * weight_scale)) / 128)));
		input1[i] = static_cast<uint8_t>(std::max(0, std::min(127, sum >> weight_scale)));
	}

	//Material
	int8_t* mat_ptr = w0 + 15 * Layer0_PaddedInputDimensions;
	std::int32_t material = b0[15];
	for (int j = 0; j < Layer0_InputDimensions; ++j) {
		material += mat_ptr[j] * input[j];
	}
	material = (material * 600 * 16) / (127 * (1 << weight_scale)); //Scaling

	//Positional
	int positional = *b2;
	for (int i = 0; i < 32; ++i) {
		const int offset = i * Layer1_PaddedInputDimensions;
		std::int32_t sum = b1[i];
		for (int j = 0; j < Layer1_InputDimensions; ++j) {
			sum += w1[offset + j] * buffer[j];
		}
		positional += w2[i] * static_cast<uint8_t>(std::max(0, std::min(127, sum >> weight_scale)));
	}
	return material + positional;
}


// Reduce 8x8 accumulators into 1x8
static inline __m256i accumulator_reduce(__m256i accs[8], __m256i bias) {
	const __m256i one = _mm256_set1_epi16(1);

	accs[0] = _mm256_hadd_epi32(_mm256_madd_epi16(accs[0], one), _mm256_madd_epi16(accs[1], one));
	accs[1] = _mm256_hadd_epi32(_mm256_madd_epi16(accs[2], one), _mm256_madd_epi16(accs[3], one));
	accs[2] = _mm256_hadd_epi32(_mm256_madd_epi16(accs[4], one), _mm256_madd_epi16(accs[5], one));
	accs[3] = _mm256_hadd_epi32(_mm256_madd_epi16(accs[6], one), _mm256_madd_epi16(accs[7], one));

	//a0 a1 a2 a3; b0 b1 b2 b3; c0 c1 c2 c3; d0 d1 d2 d3; a4 a5 a6 a7; b4 b5 b6 b7; c4 c5 c6 c7; d4 d5 d6 d7
	//e0 e1 e2 e3; f0 f1 f2 f3; g0 g1 g2 g3; h0 h1 h2 h3; e4 e5 e6 e7; f4 f5 f6 f7; g4 g5 g6 g7; h4 h5 h6 h7
	//a4 a5 a6 a7; b4 b5 b6 b7; c4 c5 c6 c7; d4 d5 d6 d7; e0 e1 e2 e3; f0 f1 f2 f3; g0 g1 g2 g3; h0 h1 h2 h3
	accs[0] = _mm256_hadd_epi32(accs[0], accs[1]);
	accs[1] = _mm256_hadd_epi32(accs[2], accs[3]);
	accs[2] = _mm256_permute2x128_si256(accs[0], accs[1], 0b100001);

	//Blend and add bias
	return _mm256_add_epi32(bias, _mm256_blend_epi32(
			_mm256_add_epi32(accs[0], accs[2]),
			_mm256_add_epi32(accs[1], accs[2]),
		0b11110000));
}

// Reduce a sinlge accumulator (compilers do emit good asm here)
static inline int reduce(__m256i acc) {
	return _mm256_extract_epi32(acc, 0) + _mm256_extract_epi32(acc, 1) + _mm256_extract_epi32(acc, 2) + _mm256_extract_epi32(acc, 3) +
	_mm256_extract_epi32(acc, 4) + _mm256_extract_epi32(acc, 5) + _mm256_extract_epi32(acc, 6) + _mm256_extract_epi32(acc, 7);
}

//Packs integers that are inside a range of [0..255] into the lower and upper lane of the output
template<int sub>
static inline __m256i pack8_fromint_lane(__m256i x, __m256i y) {
	__m256i shuffle_mask;
	if constexpr (sub == 0) {
		shuffle_mask = _mm256_setr_epi8(
			0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			-1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1); 
	}
	else {
		shuffle_mask = _mm256_setr_epi8(
			-1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1,
			-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12);
	}

	// Shuffle the bytes within each 128-bit lane using the control mask
	__m256i shuffled_x = _mm256_shuffle_epi8(x, shuffle_mask);
	__m256i shuffled_y = _mm256_shuffle_epi8(y, shuffle_mask); //x0, x1, y0, y1

	// Combine the shuffled results in a new 256-bit vector by picking correct lanes
	__m256i shufa = _mm256_permute2x128_si256(shuffled_x, shuffled_y, 0b100000);
	__m256i shufb = _mm256_permute2x128_si256(shuffled_x, shuffled_y, 0b110001);

	return _mm256_or_si256(shufa, shufb);
}

static inline int propagate_avx2(const __m256i* input_simd, const __m256i* restrict w0, const __m256i* restrict b0) noexcept {
	constexpr int Layer0_PaddedInputDimensions = 1024;
	constexpr int Layer0_InputDimensions = 1024;
	constexpr int Layer1_PaddedInputDimensions = 32;
	constexpr int Layer1_InputDimensions = 30;
	constexpr int weight_scale = 6;

	const __m256i zero = _mm256_setzero_si256();
	const __m256i max = _mm256_set1_epi16(127);
	const __m256i clamp_min = _mm256_set1_epi32(-8192);
	const __m256i clamp_max = _mm256_set1_epi32(8192);
	const __m256i relu_max = _mm256_set1_epi32(127);

	__m256i sum8;
	__m256i accs[8]; //8x 8 packed int32's

	//First Iteration
	for (int i = 0; i < 8; ++i) {
		__m256i& acc = accs[i] = _mm256_setzero_si256();
		for (int m = 0; m < 32; m++) {
			acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(input_simd[m], *w0++));
		}
	}
	sum8 = _mm256_max_epi32(clamp_min, _mm256_min_epi32(clamp_max, accumulator_reduce(accs, b0[0])));
	auto input_data = pack8_fromint_lane<0>(
		_mm256_min_epi32(relu_max, _mm256_srai_epi32(_mm256_mullo_epi32(sum8, sum8), 19)), 
		_mm256_max_epi32(zero, _mm256_min_epi32(relu_max, _mm256_srai_epi32(sum8, 6))));
	
	//Second Iteration + Material bypass
	for (int i = 0; i < 8; ++i) {
		__m256i& acc = accs[i] = _mm256_setzero_si256();
		for (int m = 0; m < 32; m++) {
			acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(input_simd[m], *w0++));
		}
	}
	sum8 = accumulator_reduce(accs, b0[1]);
	int material = (_mm256_extract_epi32(sum8, 7) * 600 * 16) / (127 * (1 << 6));

	sum8 = _mm256_max_epi32(clamp_min, _mm256_min_epi32(clamp_max, sum8));
	
	input_data = _mm256_or_si256(input_data, pack8_fromint_lane<1>(
		_mm256_min_epi32(relu_max, _mm256_srai_epi32(_mm256_mullo_epi32(sum8, sum8), 19)), 
		_mm256_max_epi32(zero, _mm256_min_epi32(relu_max, _mm256_srai_epi32(sum8, 6)))));

	//reduce 8x8 accumulators via relu into single sum. Once 32 clamped [0..127] integers are calculated they are packed 
	__m256i sums[4];
	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 8; ++i) {
			accs[i] = _mm256_maddubs_epi16(input_data, *w0++);
		}
		sums[j] = _mm256_max_epi32(zero, _mm256_min_epi32(relu_max, _mm256_srai_epi32(accumulator_reduce(accs, b0[2+j]), 6)));
	}

	//Positional
	auto positional = _mm256_or_si256(pack8_fromint_lane<0>(sums[0], sums[2]), pack8_fromint_lane<1>(sums[1], sums[3]));
	positional = _mm256_madd_epi16(_mm256_maddubs_epi16(positional, *w0++), _mm256_set1_epi16(1));

	
	return material + reduce(positional);
}


static void nnue_adjust(int8_t* w0, int8_t* w1)
{
	//AVX2 Version:

	//Inputs are limited to certain max bits by relusqr layer before so the following trick works - instead of this:
	//acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(input_simd[j], pos_ptr[j]), one)); 
	//In every iteration - we can do this 32 times for 16 bit accumulator. It cannot overflow with 32 steps. But will 100% with 33 steps.
	//Perfection. We reduce avx2s 16to 32bit promotion _mm256_madd_epi16(a, one) from 32 calls to 1
	//acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(input_simd[j], pos_ptr[j]));

	//Second optimisation - work on 8 accumulators at a time. 2 iterations for all 30 relu and relusqr bits
	//This is correct for 99.98% of positions but once every few 10000 the mullo_epi32 would need more than 32bits. 
	//_mm256_srli_epi32(_mm256_mullo_epi32(sum8, sum8), 19) 
	//Solution:
	//sum*sum >> 19 is smaller than 127 (2^7) if and only if sum < 8192 - so we clamp it and dont get overflows. 
	//sum >> 6 is not impacted by this clamping.

	//Need to adjust w1 to align from 0..15..30 to 0..31
	//w1[15] has to be zero - and the weights above 15 are shifted upwards 1 place
	for (int i = 0; i < 32; ++i) {
		const int offset = i * 32;

		int prev = w1[offset + 15];
		w1[offset + 15] = 0;
		for (int j = 16; j < 32; ++j) { 
			int tmp = w1[offset + j];
			w1[offset + j] = prev;
			prev = tmp;
		}
	}

	//We could permute Layer0 weights during init to skip permute and storing lower and upper 128 bits completely!
	//optimisation into two a direct stores possible. (removal of permute and interleaved own, opp in input weights)
	//summary: removal of 2 instructions (costing 2 cpi) possible if we do some work on init and linear memory access during runtime!
	//we just create a mapping table to permute all 1024 weights now. 

	//What does it mean? Normally we store into 0+i and 512+i
	//This was needed before: _mm256_permute4x64_epi64(..._MM_SHUFFLE(3, 1, 2, 0)) to bring it into linear form
	//With below code we map the shuffled indices into 0+i and interleave them

	//So we unshuffle
	int8_t w0_tmp[1024];

	for (int w = 0; w < 16; w++) {
		int8_t* w0_start = w0;
		int8_t* s = w0_tmp;
		int8_t* lower = w0_start;
		int8_t* upper = w0_start + 512;
		for (int i = 0; i < 512; i += 16) {
			for (int m = i; m < i + 16; m++) {
				*s++ = lower[m];
			}
			for (int m = i; m < i + 16; m++) {
				*s++ = upper[m];
			}
		}
		for (int i = 0; i < 1024; i++) {
			w0_start[i] = w0_tmp[i];
		}

		__m256i* w0_simd = reinterpret_cast<__m256i*>(w0_start);
		for (int i = 0; i < 1024 / 32; i++) {
			w0_simd[i] = _mm256_permute4x64_epi64(w0_simd[i], _MM_SHUFFLE(3, 1, 2, 0));
		}
		w0 += 1024; //next layer
	}
}

// Layer shape - and hash info - calculation happens elsewhere
template <int InDims, int OutDims>
struct LayerShape {
	static constexpr int InputDimensions = InDims;
	static constexpr int PaddedInputDimensions = (InDims + 32 - 1) / 32 * 32; //Upwards round towards 32
	static constexpr int OutputDimensions = OutDims;

	using Bias = int32_t[OutputDimensions];
	using Weight = int8_t[OutputDimensions * PaddedInputDimensions];

	// Hash value embedded in the evaluation file
	static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
		std::uint32_t hashValue = 0xCC03DAE4u;
		hashValue += OutDims;
		hashValue ^= prevHash >> 1;
		hashValue ^= prevHash << 31;
		return hashValue;
	}
};

struct Network
{
  static constexpr int FC_0_OUTPUTS = 15;
  static constexpr int FC_1_OUTPUTS = 32;

  // Hash value embedded in the evaluation file
  static constexpr std::uint32_t get_hash_value() {
	std::uint32_t hash = 0xEC42E90Du;
	hash ^= 1024 * 2;
	hash = Layer0::get_hash_value(hash);
	hash += 0x538D24C7u; //consolidated clipping layers
	hash = Layer1::get_hash_value(hash);
	hash += 0x538D24C7u; //consolidated clipping layers
	hash = Layer2::get_hash_value(hash);
	return hash;
  }

	using Layer0 = LayerShape<1024, 16>;
	using Layer1 = LayerShape<30, 32>;
	using Layer2 = LayerShape<32, 1>;
	static constexpr int size_w = sizeof(Layer0::Weight) + sizeof(Layer1::Weight) + sizeof(Layer2::Weight);
	static constexpr int size_b = (sizeof(Layer0::Bias) + sizeof(Layer1::Bias) + sizeof(Layer2::Bias)) / sizeof(int32_t);
		

	//Network data
	alignas(64) int32_t bias0[size_b]; alignas(64) int8_t weight0[size_w];
	int32_t* bias1;  
	int8_t*  weight1; 
	int32_t* bias2;  
	int8_t*  weight2; 

	//Temp memory at this point
	Layers::AffineTransform<TransformedFeatureDimensions, FC_0_OUTPUTS + 1> fc_0;
	Layers::AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS> fc_1;
	Layers::AffineTransform<FC_1_OUTPUTS, 1> fc_2;


  // Read network parameters
  bool read_parameters(std::istream& stream) {
    bool ok = fc_0.read_parameters(stream)
           && fc_1.read_parameters(stream)
           && fc_2.read_parameters(stream);
    
	bias1  = bias0 + sizeof(Layer0::Bias) / sizeof(int32_t); 
	weight1 = weight0 + sizeof(Layer0::Weight);
	bias2 = bias0 + (sizeof(Layer0::Bias) + sizeof(Layer1::Bias)) / sizeof(int32_t); 
	weight2 = weight0 + sizeof(Layer0::Weight) + sizeof(Layer1::Weight);

	memcpy(weight0, fc_0.weights, sizeof(fc_0.weights)); static_assert(sizeof(fc_0.weights) == sizeof(Layer0::Weight));
	memcpy(weight1, fc_1.weights, sizeof(fc_1.weights)); static_assert(sizeof(fc_1.weights) == sizeof(Layer1::Weight));
	memcpy(weight2, fc_2.weights, sizeof(fc_2.weights)); static_assert(sizeof(fc_2.weights) == sizeof(Layer2::Weight));
	
	memcpy(bias0, fc_0.biases, sizeof(fc_0.biases)); static_assert(sizeof(fc_0.biases) == sizeof(Layer0::Bias));
	memcpy(bias1, fc_1.biases, sizeof(fc_1.biases)); static_assert(sizeof(fc_1.biases) == sizeof(Layer1::Bias));
	memcpy(bias2, fc_2.biases, sizeof(fc_2.biases)); static_assert(sizeof(fc_2.biases) == sizeof(Layer2::Bias));
	
	nnue_adjust(weight0, weight1);
	return ok;
  }

  // Write network parameters
  bool write_parameters([[maybe_unused]] std::ostream& stream) const {
    return true;
  }

  inline std::int32_t propagate2(const __m256i* input_simd)
  {
	int outputValue = propagate_avx2(input_simd,
					reinterpret_cast<const __m256i*>(weight0), 
					reinterpret_cast<const __m256i*>(bias0)) + bias2[0];

    return outputValue;
  }

  std::int32_t propagate(const TransformedFeatureType* transformedFeatures)
  {
	/*
	const __m256i zero = _mm256_setzero_si256();
	const __m256i max = _mm256_set1_epi16(127);
	const __m256i clamp_min = _mm256_set1_epi32(-8192);
	const __m256i clamp_max = _mm256_set1_epi32(8192);
	const __m256i relu_max = _mm256_set1_epi32(127);

	//Featuretransform
	for (int j = 0; j < 512 / 16; ++j) {
		auto sum0 = _mm256_max_epi16(zero, _mm256_min_epi16(max, acc[j]));
		auto sum1 = _mm256_max_epi16(zero, _mm256_min_epi16(max, acc[j + 32]));
		auto sum2 = _mm256_max_epi16(zero, _mm256_min_epi16(max, acc[j + 64]));
		auto sum3 = _mm256_max_epi16(zero, _mm256_min_epi16(max, acc[j + 96]));

		input_simd[j] = _mm256_packus_epi16(
			_mm256_srli_epi16(_mm256_mullo_epi16(sum0, sum1), 7),
			_mm256_srli_epi16(_mm256_mullo_epi16(sum2, sum3), 7));
	}


	//Pointer declarations - make code more readable and compiled away offsets in inner loops
	uint8_t* own_half = input;
	uint8_t* opp_half = input + traits / 2;
	const int16_t* own1 = acc;
	const int16_t* own2 = acc + 512;
	const int16_t* opp1 = acc + 1024;
	const int16_t* opp2 = acc + 1536;

	//Featuretransform
	for (int j = 0; j < 512; ++j) {
		BiasType sum0 = std::max<int>(0, std::min<int>(127, own1[j]));
		BiasType sum1 = std::max<int>(0, std::min<int>(127, own2[j]));
		BiasType sum3 = std::max<int>(0, std::min<int>(127, opp1[j]));
		BiasType sum4 = std::max<int>(0, std::min<int>(127, opp2[j]));
		own_half[j] = static_cast<uint8_t>(sum0 * sum1 / 128);
		opp_half[j] = static_cast<uint8_t>(sum3 * sum4 / 128);
	}
*/

	int outputValue = propagate_scalar(transformedFeatures, fc_0.weights, fc_0.biases, fc_1.weights, fc_1.biases, fc_2.weights, fc_2.biases);

	//int outputValue = propagate_avx2(reinterpret_cast<const __m256i*>(transformedFeatures),
	//				reinterpret_cast<const __m256i*>(weight0), 
	//				reinterpret_cast<const __m256i*>(bias0)) + bias2[0];

    return outputValue;
  }
};

}  // namespace Stockfish::Eval::NNUE

#endif // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
