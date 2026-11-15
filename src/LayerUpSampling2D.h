/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"

namespace beednn {
class LayerUpSampling2D : public Layer
{
public:
	explicit LayerUpSampling2D(Index iInRows, Index iInCols, Index iInChannels, Index iRowFactor = 2, Index iColFactor = 2);
	virtual ~LayerUpSampling2D() override;

	virtual Layer* clone() const override;

	virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
	virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

private:
	Index _iInRows;
	Index _iInCols;
	Index _iInChannels;
	Index _iRowFactor;
	Index _iColFactor;
};
}
