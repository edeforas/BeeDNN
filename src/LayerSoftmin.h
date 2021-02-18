/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerSoftmin_
#define LayerSoftmin_

#include "Layer.h"
#include "Matrix.h"

class LayerSoftmin : public Layer
{
public:
    LayerSoftmin();
    virtual ~LayerSoftmin() override;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;
};

#endif
