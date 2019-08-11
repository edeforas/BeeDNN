/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerDropout_
#define LayerDropout_

#include "Layer.h"
#include "Matrix.h"

#include <string>
using namespace std;

class Activation;

class LayerDropout : public Layer
{
public:
    LayerDropout(int iSize,float fRate);
    virtual ~LayerDropout() override;

    virtual Layer* clone() const override;

    virtual void init() override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) const override;

    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

    float get_rate() const;

private:
    void create_mask();

    float _fRate;
    MatrixFloat _mask;
};

#endif
