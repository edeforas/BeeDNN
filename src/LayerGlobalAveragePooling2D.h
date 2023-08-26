/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"

class LayerGlobalAveragePooling2D : public Layer
{
public:
	explicit LayerGlobalAveragePooling2D(Index iInRows, Index iInCols, Index iInChannels);
    virtual ~LayerGlobalAveragePooling2D() override;

	void get_params(Index& iInRows, Index& iInCols, Index& iInChannels) const;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

private:
	Index _iInRows;
	Index _iInCols;
	Index _iInChannels;
	Index _iInPlaneSize;

	float _fInvKernelSize;
};

