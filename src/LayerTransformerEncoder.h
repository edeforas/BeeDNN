/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"

#include <vector>

namespace beednn {
class LayerTransformerEncoder : public Layer
{
public:
	LayerTransformerEncoder(Index iSequenceLength, Index iModelSize, Index iFeedForwardSize, Index iHeads, float fAttentionDropoutRate = 0.f);
	virtual ~LayerTransformerEncoder() override;

	virtual Layer* clone() const override;
	virtual void init() override;

	void set_attention_dropout_rate(float fRate);
	float attention_dropout_rate() const;
	void set_causal_mask(bool bEnabled);
	bool causal_mask() const;
	// Nonzero entries allow query-key attention; the same mask applies to each sample and head.
	void set_attention_mask(const MatrixFloat& mAllowed);
	void clear_attention_mask();

	virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
	virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

private:
	struct ForwardCache
	{
		std::vector<float> input;
		std::vector<float> query;
		std::vector<float> key;
		std::vector<float> value;
		std::vector<float> attention;
		std::vector<float> attentionDropoutMask;
		std::vector<float> context;
		std::vector<float> normalized1;
		std::vector<float> normOutput1;
		std::vector<float> inverseStd1;
		std::vector<float> feedForwardPre;
		std::vector<float> feedForward;
		std::vector<float> feedForwardOutput;
		std::vector<float> normalized2;
		std::vector<float> inverseStd2;
	};

	bool attention_allowed(Index iQueryToken, Index iKeyToken) const;
	void validate_attention_mask() const;
	void forward_sample(const float* pIn, float* pOut, const float* pDropoutMask, ForwardCache& cache) const;

	Index _iSequenceLength;
	Index _iModelSize;
	Index _iFeedForwardSize;
	Index _iHeads;
	float _fAttentionDropoutRate;
	bool _bCausalMask;
	std::vector<unsigned char> _attentionMask;
	MatrixFloat _mAttentionDropoutMask;
};
}
