/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerGLU_
#define LayerGLU_

#include "Layer.h"
#include "Matrix.h"

class Activation;

class LayerGLU : public Layer
{
public:
    explicit LayerGLU();
    virtual ~LayerGLU() override;

    virtual Layer* clone() const override;

    virtual void init() override;
	
	virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
	virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

private:
	Activation* _pActivation;

};

#endif
