/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"
#include "Layer.h"

#include <vector>

class LayerRNN : public Layer
{
public:
    explicit LayerRNN(int iFrameSize,int iUnits);
    virtual ~LayerRNN();
    virtual void init() override;

    virtual Layer* clone() const override =0;
    virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
    virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

    virtual void forward_frame(const MatrixFloat& mInFrame, MatrixFloat& mOut) =0;
    virtual void backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mH, const MatrixFloat& mHm1,const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) =0;

protected:
    MatrixFloat _h;
    std::vector<MatrixFloat> _savedH; // used for back propagation

    int _iFrameSize;
    int _iUnits;
};

