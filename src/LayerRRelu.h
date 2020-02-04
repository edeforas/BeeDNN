/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerRRelu_
#define LayerRRelu_

#include "Layer.h"
#include "Matrix.h"

class LayerRRelu : public Layer
{
public:
    explicit LayerRRelu(float alpha1=8.f, float alpha2=3.f);
    virtual ~LayerRRelu() override;

    virtual Layer* clone() const override;

    virtual void init() override;
	
	virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
	virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

	void get_params(float& alpha1, float& alpha2);

private:
	MatrixFloat _slopes;
	float _alpha1;
	float _alpha2;
	float _invAlpha1;
	float _invAlpha2;
	float _invAlphaMean;	
};

#endif
