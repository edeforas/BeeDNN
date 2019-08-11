/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerSoftmax_
#define LayerSoftmax_

#include "Layer.h"
#include "Matrix.h"

#include <string>
using namespace std;

class Activation;

class LayerSoftmax : public Layer
{
public:
    LayerSoftmax();
    virtual ~LayerSoftmax() override;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) const override;
	
    virtual void backpropagation(const MatrixFloat &mInput,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

};

#endif
