/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerPoolMax1D_
#define LayerPoolMax1D_

#include "Layer.h"
#include "Matrix.h"

class LayerPoolMax1D : public Layer
{
public:
    explicit LayerPoolMax1D(int iInSize,int iWindowSize);
    virtual ~LayerPoolMax1D() override;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;
};

#endif
