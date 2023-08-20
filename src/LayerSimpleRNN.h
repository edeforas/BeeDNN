/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerSimpleRNN_
#define LayerSimpleRNN_

#include "Layer.h"
#include "Matrix.h"
#include "LayerRNN.h"

// Simple RNN algorithm as in : https://arxiv.org/abs/1610.02583
namespace bee {
class LayerSimpleRNN : public LayerRNN
{
public:
    explicit LayerSimpleRNN(int iSampleSize,int iUnits);
    virtual ~LayerSimpleRNN();
    virtual void init() override;

    virtual bee::Layer* clone() const override;
    virtual void forward_frame(const MatrixFloat& mIn, MatrixFloat& mOut) override;

    virtual void backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mH, const MatrixFloat& mHm1, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

private:
    MatrixFloat _whh, _wxh, _bh;
};
}
#endif
