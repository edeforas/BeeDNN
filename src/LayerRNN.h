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
#include "Layer.h"

class LayerRNN : public Layer
{
public:
    explicit LayerRNN(int iSampleSize,int iUnits);
    virtual ~LayerRNN();
    virtual void init();

    virtual Layer* clone() const =0;
    virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
    virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

    virtual void forward_sample(const MatrixFloat& mIn, MatrixFloat& mOut) =0;
    virtual void backpropagation_sample(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) =0;

protected:
    int _iSampleSize;
    int _iUnits;
};

#endif