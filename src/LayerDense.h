/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerDense_
#define LayerDense_

#include "Layer.h"
#include "Matrix.h"

class LayerDense : public Layer
{
public:
    LayerDense(int iInSize,int iOutSize,bool bHasBias);
    virtual ~LayerDense() override;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mMatin, MatrixFloat &mMatOut) const override;

    virtual void init() override;
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta) override;

    bool has_bias() const;

private:
    bool _bHasBias;
};

#endif
