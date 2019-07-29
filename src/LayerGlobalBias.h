/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerGlobalBias_
#define LayerGlobalBias_

#include "Layer.h"
#include "Matrix.h"

class LayerGlobalBias : public Layer
{
public:
    LayerGlobalBias();
    virtual ~LayerGlobalBias();

    virtual Layer* clone() const override;

    virtual void init() override;
    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const override;

    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta) override;

    float gain() const;

    virtual bool has_weight() override;
    virtual MatrixFloat& weights() override;
    virtual MatrixFloat& gradient_weights() override;

private:
    MatrixFloat _weight,_mDeltaWeight;
};

#endif
