/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerSimplestRNN_
#define LayerSimplestRNN_

#include "Layer.h"
#include "Matrix.h"
#include "LayerRNN.h"

// Simplest possible RNN algorithm (removed the time distributed applied on the input)
// this layer is simpler than the LayerSimpleRNN

class LayerSimplestRNN : public LayerRNN
{
public:
    explicit LayerSimplestRNN(int iSampleSize);
    virtual ~LayerSimplestRNN();
    virtual void init() override;

    virtual Layer* clone() const override;
    virtual void forward_frame(const MatrixFloat& mInFrame, MatrixFloat& mOut) override;
    virtual void backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

private:
    MatrixFloat _whh, _h; // todo move _h to super class ?
};

#endif
