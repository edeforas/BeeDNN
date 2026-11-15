/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerTransformerEncoder.h"

#include <algorithm>
#include <cassert>
#include <cmath>

using namespace std;
namespace beednn {
namespace {

void layer_norm_forward(const vector<float>& input, vector<float>& normalized, vector<float>& output,
	vector<float>& inverseStd, Index sequenceLength, Index modelSize, const float* gamma, const float* beta)
{
	const float epsilon = 1.e-5f;
	normalized.resize(sequenceLength * modelSize);
	output.resize(sequenceLength * modelSize);
	inverseStd.resize(sequenceLength);

	for (Index token = 0; token < sequenceLength; token++)
	{
		const Index offset = token * modelSize;
		float mean = 0.f;
		for (Index feature = 0; feature < modelSize; feature++)
			mean += input[offset + feature];
		mean /= (float)modelSize;

		float variance = 0.f;
		for (Index feature = 0; feature < modelSize; feature++)
		{
			const float centered = input[offset + feature] - mean;
			variance += centered * centered;
		}
		variance /= (float)modelSize;
		inverseStd[token] = 1.f / sqrtf(variance + epsilon);

		for (Index feature = 0; feature < modelSize; feature++)
		{
			const float norm = (input[offset + feature] - mean) * inverseStd[token];
			normalized[offset + feature] = norm;
			output[offset + feature] = gamma[feature] * norm + beta[feature];
		}
	}
}

void layer_norm_backward(const float* gradientOut, const vector<float>& normalized, const vector<float>& inverseStd,
	float* gradientGamma, float* gradientBeta, float* gradientIn, Index sequenceLength, Index modelSize, const float* gamma)
{
	for (Index token = 0; token < sequenceLength; token++)
	{
		const Index offset = token * modelSize;
		float sumGradient = 0.f;
		float sumGradientNorm = 0.f;
		for (Index feature = 0; feature < modelSize; feature++)
		{
			const float gradient = gradientOut[offset + feature] * gamma[feature];
			sumGradient += gradient;
			sumGradientNorm += gradient * normalized[offset + feature];
			gradientGamma[feature] += gradientOut[offset + feature] * normalized[offset + feature];
			gradientBeta[feature] += gradientOut[offset + feature];
		}

		const float scale = inverseStd[token] / (float)modelSize;
		for (Index feature = 0; feature < modelSize; feature++)
		{
			const float gradient = gradientOut[offset + feature] * gamma[feature];
			gradientIn[offset + feature] = scale *
				((float)modelSize * gradient - sumGradient - normalized[offset + feature] * sumGradientNorm);
		}
	}
}
}

LayerTransformerEncoder::LayerTransformerEncoder(Index iSequenceLength, Index iModelSize, Index iFeedForwardSize, Index iHeads, float fAttentionDropoutRate) :
	Layer("TransformerEncoder"),
	_iSequenceLength(iSequenceLength),
	_iModelSize(iModelSize),
	_iFeedForwardSize(iFeedForwardSize),
	_iHeads(iHeads),
	_fAttentionDropoutRate(fAttentionDropoutRate),
	_bCausalMask(false)
{
	assert(_iSequenceLength > 0);
	assert(_iModelSize > 0);
	assert(_iFeedForwardSize > 0);
	assert(_iHeads > 0);
	assert(_iModelSize % _iHeads == 0);
	assert(_fAttentionDropoutRate >= 0.f && _fAttentionDropoutRate < 1.f);

	set_weight_initializer("GlorotUniform");
	set_bias_initializer("Zeros");
	init();
}

LayerTransformerEncoder::~LayerTransformerEncoder()
{ }

Layer* LayerTransformerEncoder::clone() const
{
	LayerTransformerEncoder* pLayer = new LayerTransformerEncoder(
		_iSequenceLength, _iModelSize, _iFeedForwardSize, _iHeads, _fAttentionDropoutRate);
	pLayer->_weight = _weight;
	pLayer->_bias = _bias;
	pLayer->_attentionMask = _attentionMask;
	pLayer->_bCausalMask = _bCausalMask;
	return pLayer;
}

void LayerTransformerEncoder::set_attention_dropout_rate(float fRate)
{
	assert(fRate >= 0.f && fRate < 1.f);
	_fAttentionDropoutRate = fRate;
}

float LayerTransformerEncoder::attention_dropout_rate() const
{
	return _fAttentionDropoutRate;
}

void LayerTransformerEncoder::set_causal_mask(bool bEnabled)
{
	_bCausalMask = bEnabled;
	validate_attention_mask();
}

bool LayerTransformerEncoder::causal_mask() const
{
	return _bCausalMask;
}

void LayerTransformerEncoder::set_attention_mask(const MatrixFloat& mAllowed)
{
	assert(mAllowed.rows() == _iSequenceLength);
	assert(mAllowed.cols() == _iSequenceLength);
	_attentionMask.resize(_iSequenceLength * _iSequenceLength);
	for (Index query = 0; query < _iSequenceLength; query++)
	{
		for (Index key = 0; key < _iSequenceLength; key++)
			_attentionMask[query * _iSequenceLength + key] = mAllowed(query, key) > 0.f ? 1 : 0;
	}
	validate_attention_mask();
}

void LayerTransformerEncoder::clear_attention_mask()
{
	_attentionMask.clear();
}

bool LayerTransformerEncoder::attention_allowed(Index iQueryToken, Index iKeyToken) const
{
	const bool allowedByCustomMask = _attentionMask.empty() ||
		_attentionMask[iQueryToken * _iSequenceLength + iKeyToken] != 0;
	const bool allowedByCausalMask = !_bCausalMask || iKeyToken <= iQueryToken;
	return allowedByCustomMask && allowedByCausalMask;
}

void LayerTransformerEncoder::validate_attention_mask() const
{
	for (Index query = 0; query < _iSequenceLength; query++)
	{
		bool hasAllowedKey = false;
		for (Index key = 0; key < _iSequenceLength; key++)
			hasAllowedKey = hasAllowedKey || attention_allowed(query, key);
		assert(hasAllowedKey);
	}
}

void LayerTransformerEncoder::init()
{
	const Index modelSizeSquared = _iModelSize * _iModelSize;
	const Index feedForwardWeights = _iModelSize * _iFeedForwardSize;
	const Index weightCount = 4 * modelSizeSquared + 2 * feedForwardWeights;
	const Index biasCount = 9 * _iModelSize + _iFeedForwardSize;
	_weight.resize(1, weightCount);
	_gradientWeight.setZero(1, weightCount);
	_bias.setZero(1, biasCount);
	_gradientBias.setZero(1, biasCount);

	setRandomUniform(_weight, -1.f, 1.f);
	float* pWeight = _weight.data();
	const float projectionScale = sqrtf(6.f / (2.f * _iModelSize));
	const float feedForwardScale = sqrtf(6.f / (_iModelSize + _iFeedForwardSize));
	for (Index i = 0; i < 4 * modelSizeSquared; i++)
		pWeight[i] *= projectionScale;
	for (Index i = 4 * modelSizeSquared; i < weightCount; i++)
		pWeight[i] *= feedForwardScale;

	const Index gamma1Offset = 5 * _iModelSize + _iFeedForwardSize;
	const Index gamma2Offset = 7 * _iModelSize + _iFeedForwardSize;
	for (Index i = 0; i < _iModelSize; i++)
	{
		_bias(gamma1Offset + i) = 1.f;
		_bias(gamma2Offset + i) = 1.f;
	}
	Layer::init();
}

void LayerTransformerEncoder::forward_sample(const float* pIn, float* pOut, const float* pDropoutMask, ForwardCache& cache) const
{
	const Index sequenceLength = _iSequenceLength;
	const Index modelSize = _iModelSize;
	const Index feedForwardSize = _iFeedForwardSize;
	const Index headSize = modelSize / _iHeads;
	const Index modelSizeSquared = modelSize * modelSize;
	const Index feedForwardWeightOffset = 4 * modelSizeSquared;
	const Index feedForwardOutputWeightOffset = feedForwardWeightOffset + modelSize * feedForwardSize;
	const float* pWeight = _weight.data();
	const float* pBias = _bias.data();
	const Index keyBiasOffset = modelSize;
	const Index valueBiasOffset = 2 * modelSize;
	const Index outputBiasOffset = 3 * modelSize;
	const Index feedForwardBiasOffset = 4 * modelSize;
	const Index feedForwardOutputBiasOffset = 4 * modelSize + feedForwardSize;
	const Index gamma1Offset = 5 * modelSize + feedForwardSize;
	const Index beta1Offset = 6 * modelSize + feedForwardSize;
	const Index gamma2Offset = 7 * modelSize + feedForwardSize;
	const Index beta2Offset = 8 * modelSize + feedForwardSize;

	cache.input.resize(sequenceLength * modelSize);
	cache.query.resize(sequenceLength * modelSize);
	cache.key.resize(sequenceLength * modelSize);
	cache.value.resize(sequenceLength * modelSize);
	cache.attention.resize(_iHeads * sequenceLength * sequenceLength);
	cache.attentionDropoutMask.resize(_iHeads * sequenceLength * sequenceLength);
	cache.context.assign(sequenceLength * modelSize, 0.f);
	vector<float> residual1(sequenceLength * modelSize);

	const Index attentionSize = _iHeads * sequenceLength * sequenceLength;
	if (_bTrainMode && _fAttentionDropoutRate > 0.f)
	{
		if (pDropoutMask != nullptr)
		{
			copy(pDropoutMask, pDropoutMask + attentionSize, cache.attentionDropoutMask.begin());
		}
		else
		{
			uniform_real_distribution<float> distribution(0.f, 1.f);
			default_random_engine& engine = randomEngine();
			const float keepScale = 1.f / (1.f - _fAttentionDropoutRate);
			for (Index i = 0; i < attentionSize; i++)
				cache.attentionDropoutMask[i] = distribution(engine) >= _fAttentionDropoutRate ? keepScale : 0.f;
		}
	}
	else
	{
		fill(cache.attentionDropoutMask.begin(), cache.attentionDropoutMask.end(), 1.f);
	}

	for (Index token = 0; token < sequenceLength; token++)
	{
		for (Index feature = 0; feature < modelSize; feature++)
		{
			const float exponent = (float)(2 * (feature / 2)) / (float)modelSize;
			const float angle = (float)token / powf(10000.f, exponent);
			const float position = (feature % 2 == 0) ? sinf(angle) : cosf(angle);
			cache.input[token * modelSize + feature] = pIn[token * modelSize + feature] + position;
		}
	}

	for (Index token = 0; token < sequenceLength; token++)
	{
		for (Index output = 0; output < modelSize; output++)
		{
			float query = pBias[output];
			float key = pBias[keyBiasOffset + output];
			float value = pBias[valueBiasOffset + output];
			for (Index input = 0; input < modelSize; input++)
			{
				const float feature = cache.input[token * modelSize + input];
				query += feature * pWeight[input * modelSize + output];
				key += feature * pWeight[modelSizeSquared + input * modelSize + output];
				value += feature * pWeight[2 * modelSizeSquared + input * modelSize + output];
			}
			cache.query[token * modelSize + output] = query;
			cache.key[token * modelSize + output] = key;
			cache.value[token * modelSize + output] = value;
		}
	}

	const float attentionScale = 1.f / sqrtf((float)headSize);
	for (Index head = 0; head < _iHeads; head++)
	{
		const Index headOffset = head * headSize;
		for (Index queryToken = 0; queryToken < sequenceLength; queryToken++)
		{
			const Index attentionRow = (head * sequenceLength + queryToken) * sequenceLength;
			float maxScore = -1.e38f;
			for (Index keyToken = 0; keyToken < sequenceLength; keyToken++)
			{
				if (!attention_allowed(queryToken, keyToken))
				{
					cache.attention[attentionRow + keyToken] = -1.e38f;
					continue;
				}

				float score = 0.f;
				for (Index feature = 0; feature < headSize; feature++)
				{
					score += cache.query[queryToken * modelSize + headOffset + feature] *
						cache.key[keyToken * modelSize + headOffset + feature];
				}
				score *= attentionScale;
				cache.attention[attentionRow + keyToken] = score;
				maxScore = max(maxScore, score);
			}

			float sum = 0.f;
			for (Index keyToken = 0; keyToken < sequenceLength; keyToken++)
			{
				float& probability = cache.attention[attentionRow + keyToken];
				if (attention_allowed(queryToken, keyToken))
					probability = expf(probability - maxScore);
				else
					probability = 0.f;
				sum += probability;
			}
			for (Index keyToken = 0; keyToken < sequenceLength; keyToken++)
				cache.attention[attentionRow + keyToken] /= sum;
		}
	}

	for (Index head = 0; head < _iHeads; head++)
	{
		const Index headOffset = head * headSize;
		for (Index queryToken = 0; queryToken < sequenceLength; queryToken++)
		{
			const Index attentionRow = (head * sequenceLength + queryToken) * sequenceLength;
			for (Index keyToken = 0; keyToken < sequenceLength; keyToken++)
			{
				const Index attentionIndex = attentionRow + keyToken;
				const float probability = cache.attention[attentionIndex] * cache.attentionDropoutMask[attentionIndex];
				for (Index feature = 0; feature < headSize; feature++)
				{
					cache.context[queryToken * modelSize + headOffset + feature] +=
						probability * cache.value[keyToken * modelSize + headOffset + feature];
				}
			}
		}
	}

	vector<float> attentionOutput(sequenceLength * modelSize);
	for (Index token = 0; token < sequenceLength; token++)
	{
		for (Index output = 0; output < modelSize; output++)
		{
			float value = pBias[outputBiasOffset + output];
			for (Index input = 0; input < modelSize; input++)
				value += cache.context[token * modelSize + input] * pWeight[3 * modelSizeSquared + input * modelSize + output];
			attentionOutput[token * modelSize + output] = value;
			residual1[token * modelSize + output] = cache.input[token * modelSize + output] + value;
		}
	}

	layer_norm_forward(residual1, cache.normalized1, cache.normOutput1, cache.inverseStd1,
		sequenceLength, modelSize, pBias + gamma1Offset, pBias + beta1Offset);
	cache.feedForwardPre.resize(sequenceLength * feedForwardSize);
	cache.feedForward.resize(sequenceLength * feedForwardSize);
	cache.feedForwardOutput.resize(sequenceLength * modelSize);
	for (Index token = 0; token < sequenceLength; token++)
	{
		for (Index output = 0; output < feedForwardSize; output++)
		{
			float value = pBias[feedForwardBiasOffset + output];
			for (Index input = 0; input < modelSize; input++)
				value += cache.normOutput1[token * modelSize + input] *
					pWeight[feedForwardWeightOffset + input * feedForwardSize + output];
			cache.feedForwardPre[token * feedForwardSize + output] = value;
			cache.feedForward[token * feedForwardSize + output] = max(0.f, value);
		}
		for (Index output = 0; output < modelSize; output++)
		{
			float value = pBias[feedForwardOutputBiasOffset + output];
			for (Index input = 0; input < feedForwardSize; input++)
				value += cache.feedForward[token * feedForwardSize + input] *
					pWeight[feedForwardOutputWeightOffset + input * modelSize + output];
			cache.feedForwardOutput[token * modelSize + output] = value;
		}
	}

	vector<float> residual2(sequenceLength * modelSize);
	for (Index i = 0; i < sequenceLength * modelSize; i++)
		residual2[i] = cache.normOutput1[i] + cache.feedForwardOutput[i];
	vector<float> normOutput2;
	layer_norm_forward(residual2, cache.normalized2, normOutput2, cache.inverseStd2,
		sequenceLength, modelSize, pBias + gamma2Offset, pBias + beta2Offset);
	if (pOut != nullptr)
	{
		for (Index i = 0; i < sequenceLength * modelSize; i++)
			pOut[i] = normOutput2[i];
	}
}

void LayerTransformerEncoder::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	assert(mIn.cols() == _iSequenceLength * _iModelSize);
	mOut.resizeLike(mIn);
	const Index attentionSize = _iHeads * _iSequenceLength * _iSequenceLength;
	if (_bTrainMode && _fAttentionDropoutRate > 0.f)
		_mAttentionDropoutMask.resize(mIn.rows(), attentionSize);
	else
		_mAttentionDropoutMask.resize(0, 0);

	for (Index sample = 0; sample < mIn.rows(); sample++)
	{
		ForwardCache cache;
		forward_sample(mIn.row(sample).data(), mOut.row(sample).data(), nullptr, cache);
		if (_bTrainMode && _fAttentionDropoutRate > 0.f)
		{
			copy(cache.attentionDropoutMask.begin(), cache.attentionDropoutMask.end(),
				_mAttentionDropoutMask.row(sample).data());
		}
	}
}

void LayerTransformerEncoder::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
	assert(mIn.cols() == _iSequenceLength * _iModelSize);
	assert(mGradientOut.rows() == mIn.rows());
	assert(mGradientOut.cols() == mIn.cols());

	const Index sequenceLength = _iSequenceLength;
	const Index modelSize = _iModelSize;
	const Index feedForwardSize = _iFeedForwardSize;
	const Index headSize = modelSize / _iHeads;
	const Index modelSizeSquared = modelSize * modelSize;
	const Index feedForwardWeightOffset = 4 * modelSizeSquared;
	const Index feedForwardOutputWeightOffset = feedForwardWeightOffset + modelSize * feedForwardSize;
	const float* pWeight = _weight.data();
	const float* pBias = _bias.data();
	float* pGradientWeight = _gradientWeight.data();
	float* pGradientBias = _gradientBias.data();
	const Index keyBiasOffset = modelSize;
	const Index valueBiasOffset = 2 * modelSize;
	const Index outputBiasOffset = 3 * modelSize;
	const Index feedForwardBiasOffset = 4 * modelSize;
	const Index feedForwardOutputBiasOffset = 4 * modelSize + feedForwardSize;
	const Index gamma1Offset = 5 * modelSize + feedForwardSize;
	const Index beta1Offset = 6 * modelSize + feedForwardSize;
	const Index gamma2Offset = 7 * modelSize + feedForwardSize;
	const Index beta2Offset = 8 * modelSize + feedForwardSize;
	const float attentionScale = 1.f / sqrtf((float)headSize);

	_gradientWeight.setZero();
	_gradientBias.setZero();
	mGradientIn.setZero(mIn.rows(), mIn.cols());
	if (_bTrainMode && _fAttentionDropoutRate > 0.f)
	{
		assert(_mAttentionDropoutMask.rows() == mIn.rows());
		assert(_mAttentionDropoutMask.cols() == _iHeads * sequenceLength * sequenceLength);
	}

	for (Index sample = 0; sample < mIn.rows(); sample++)
	{
		ForwardCache cache;
		const float* pDropoutMask = (_bTrainMode && _fAttentionDropoutRate > 0.f) ?
			_mAttentionDropoutMask.row(sample).data() : nullptr;
		forward_sample(mIn.row(sample).data(), nullptr, pDropoutMask, cache);

		vector<float> gradientResidual2(sequenceLength * modelSize);
		layer_norm_backward(mGradientOut.row(sample).data(), cache.normalized2, cache.inverseStd2,
			pGradientBias + gamma2Offset, pGradientBias + beta2Offset, gradientResidual2.data(),
			sequenceLength, modelSize, pBias + gamma2Offset);
		for (Index token = 0; token < sequenceLength; token++)
		{
			for (Index output = 0; output < modelSize; output++)
			{
				const float gradient = gradientResidual2[token * modelSize + output];
				pGradientBias[feedForwardOutputBiasOffset + output] += gradient;
				for (Index input = 0; input < feedForwardSize; input++)
					pGradientWeight[feedForwardOutputWeightOffset + input * modelSize + output] +=
						cache.feedForward[token * feedForwardSize + input] * gradient;
			}
		}

		vector<float> gradientFeedForward(sequenceLength * feedForwardSize, 0.f);
		vector<float> gradientNormOutput1(sequenceLength * modelSize);
		vector<float> gradientFeedForwardPre(sequenceLength * feedForwardSize);
		for (Index token = 0; token < sequenceLength; token++)
		{
			for (Index input = 0; input < feedForwardSize; input++)
			{
				float gradient = 0.f;
				for (Index output = 0; output < modelSize; output++)
				{
					const float outputGradient = gradientResidual2[token * modelSize + output];
					gradient += outputGradient * pWeight[feedForwardOutputWeightOffset + input * modelSize + output];
				}
				gradientFeedForward[token * feedForwardSize + input] =
					cache.feedForwardPre[token * feedForwardSize + input] > 0.f ? gradient : 0.f;
				pGradientBias[feedForwardBiasOffset + input] += gradientFeedForward[token * feedForwardSize + input];
				for (Index output = 0; output < modelSize; output++)
					pGradientWeight[feedForwardWeightOffset + output * feedForwardSize + input] +=
						cache.normOutput1[token * modelSize + output] * gradientFeedForward[token * feedForwardSize + input];
			}

			for (Index input = 0; input < modelSize; input++)
			{
				float gradient = gradientResidual2[token * modelSize + input];
				for (Index output = 0; output < feedForwardSize; output++)
					gradient += gradientFeedForward[token * feedForwardSize + output] *
						pWeight[feedForwardWeightOffset + input * feedForwardSize + output];
				gradientNormOutput1[token * modelSize + input] = gradient;
			}
		}

		vector<float> gradientResidual1(sequenceLength * modelSize);
		layer_norm_backward(gradientNormOutput1.data(), cache.normalized1, cache.inverseStd1,
			pGradientBias + gamma1Offset, pGradientBias + beta1Offset, gradientResidual1.data(),
			sequenceLength, modelSize, pBias + gamma1Offset);

		vector<float> gradientContext(sequenceLength * modelSize, 0.f);
		vector<float> gradientQuery(sequenceLength * modelSize, 0.f);
		vector<float> gradientKey(sequenceLength * modelSize, 0.f);
		vector<float> gradientValue(sequenceLength * modelSize, 0.f);
		for (Index token = 0; token < sequenceLength; token++)
		{
			for (Index output = 0; output < modelSize; output++)
			{
				const float gradient = gradientResidual1[token * modelSize + output];
				pGradientBias[outputBiasOffset + output] += gradient;
				for (Index input = 0; input < modelSize; input++)
				{
					pGradientWeight[3 * modelSizeSquared + input * modelSize + output] +=
						cache.context[token * modelSize + input] * gradient;
					gradientContext[token * modelSize + input] +=
						gradient * pWeight[3 * modelSizeSquared + input * modelSize + output];
				}
			}
		}

		vector<float> gradientAttention(_iHeads * sequenceLength * sequenceLength);
		for (Index head = 0; head < _iHeads; head++)
		{
			const Index headOffset = head * headSize;
			for (Index queryToken = 0; queryToken < sequenceLength; queryToken++)
			{
				const Index attentionRow = (head * sequenceLength + queryToken) * sequenceLength;
				float softmaxDot = 0.f;
				for (Index keyToken = 0; keyToken < sequenceLength; keyToken++)
				{
					float gradient = 0.f;
					for (Index feature = 0; feature < headSize; feature++)
					{
						const Index index = headOffset + feature;
						gradient += gradientContext[queryToken * modelSize + index] *
							cache.value[keyToken * modelSize + index];
						gradientValue[keyToken * modelSize + index] +=
							cache.attention[attentionRow + keyToken] * cache.attentionDropoutMask[attentionRow + keyToken] *
							gradientContext[queryToken * modelSize + index];
					}
					gradient *= cache.attentionDropoutMask[attentionRow + keyToken];
					gradientAttention[attentionRow + keyToken] = gradient;
					softmaxDot += gradient * cache.attention[attentionRow + keyToken];
				}

				for (Index keyToken = 0; keyToken < sequenceLength; keyToken++)
				{
					const float gradientScore = cache.attention[attentionRow + keyToken] *
						(gradientAttention[attentionRow + keyToken] - softmaxDot) * attentionScale;
					for (Index feature = 0; feature < headSize; feature++)
					{
						const Index index = headOffset + feature;
						gradientQuery[queryToken * modelSize + index] +=
							gradientScore * cache.key[keyToken * modelSize + index];
						gradientKey[keyToken * modelSize + index] +=
							gradientScore * cache.query[queryToken * modelSize + index];
					}
				}
			}
		}

		float* pGradientInput = mGradientIn.row(sample).data();
		for (Index token = 0; token < sequenceLength; token++)
		{
			for (Index input = 0; input < modelSize; input++)
			{
				float gradient = gradientResidual1[token * modelSize + input];
				for (Index output = 0; output < modelSize; output++)
				{
					const Index weightIndex = input * modelSize + output;
					const float queryGradient = gradientQuery[token * modelSize + output];
					const float keyGradient = gradientKey[token * modelSize + output];
					const float valueGradient = gradientValue[token * modelSize + output];
					pGradientWeight[weightIndex] += cache.input[token * modelSize + input] * queryGradient;
					pGradientWeight[modelSizeSquared + weightIndex] += cache.input[token * modelSize + input] * keyGradient;
					pGradientWeight[2 * modelSizeSquared + weightIndex] += cache.input[token * modelSize + input] * valueGradient;
					gradient += queryGradient * pWeight[weightIndex] +
						keyGradient * pWeight[modelSizeSquared + weightIndex] +
						valueGradient * pWeight[2 * modelSizeSquared + weightIndex];
				}
				pGradientInput[token * modelSize + input] = gradient;
			}
		}

		for (Index token = 0; token < sequenceLength; token++)
		{
			for (Index output = 0; output < modelSize; output++)
			{
				pGradientBias[output] += gradientQuery[token * modelSize + output];
				pGradientBias[keyBiasOffset + output] += gradientKey[token * modelSize + output];
				pGradientBias[valueBiasOffset + output] += gradientValue[token * modelSize + output];
			}
		}
	}

	const float batchScale = 1.f / (float)mIn.rows();
	_gradientWeight *= batchScale;
	_gradientBias *= batchScale;
}
}
