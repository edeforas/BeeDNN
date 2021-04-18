/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerRNN_
#define LayerRNN_

#include "Layer.h"
#include "Matrix.h"

class LayerRNN : public Layer
{
public:
    explicit LayerRNN(const string& sType);
    virtual ~LayerRNN();

    virtual Layer* clone() const =0;

    virtual void init() override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void forward_one_sample(const MatrixFloat& mIn, MatrixFloat& mOut);
    virtual void step(const MatrixFloat& mIn, MatrixFloat& mOut) =0;

    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;
};

#endif
