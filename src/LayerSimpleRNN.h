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

// RNN algorithm as in : https://arxiv.org/abs/1610.02583

class LayerSimpleRNN : public LayerRNN
{
public:
    explicit LayerSimpleRNN(int iSampleSize,int iUnits);
    virtual ~LayerSimpleRNN();

    virtual Layer* clone() const override;
    
    virtual void init() override;

    virtual void step(const MatrixFloat& mIn, MatrixFloat& mOut) override;

private:
    int _iSampleSize;
    int _iUnits;
    MatrixFloat _whh, _wxh, _bh,_h;
};

#endif
