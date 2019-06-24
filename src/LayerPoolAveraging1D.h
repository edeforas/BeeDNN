/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerPoolAveraging1D_
#define LayerPoolAveraging1D_

#include "Layer.h"
#include "Matrix.h"

class LayerPoolAveraging1D : public Layer
{
public:
    LayerPoolAveraging1D(int iInSize,int iWindowSize);
    virtual ~LayerPoolAveraging1D() override;

    virtual Layer* clone() const override;
    virtual void init() override;

    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const override;
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta) override;

private:
    MatrixFloat _weight;
};

#endif
