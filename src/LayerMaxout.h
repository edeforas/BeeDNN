/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// Maxout network layer as in : https://arxiv.org/pdf/1302.4389.pdf

#ifndef LayerMaxout_
#define LayerMaxout_

#include "Layer.h"
#include "Matrix.h"

class LayerMaxout : public Layer
{
public:
	explicit LayerMaxout(Index iInSize, Index iOutSize);
    virtual ~LayerMaxout() override;

    virtual Layer* clone() const override;

    virtual void predict(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

private:
	Index _iInSize;
	Index _iOutSize;
	Index _iReduction;

	MatrixFloat _mMaxIndex;
};

#endif
