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
	LayerTransformerEncoder(Index iSequenceLength, Index iModelSize, Index iFeedForwardSize, Index iHeads);
	virtual ~LayerTransformerEncoder() override;

	virtual Layer* clone() const override;
	virtual void init() override;

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

	void forward_sample(const float* pIn, float* pOut, ForwardCache& cache) const;

	Index _iSequenceLength;
	Index _iModelSize;
	Index _iFeedForwardSize;
	Index _iHeads;
};
}
