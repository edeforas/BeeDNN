/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalBias.h"

///////////////////////////////////////////////////////////////////////////////
LayerGlobalBias::LayerGlobalBias() :
    Layer(0, 0, "GlobalBias")
{
    _weight.resize(1,1);
	_gradientWeight.resize(1, 1);
    LayerGlobalBias::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalBias::~LayerGlobalBias()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalBias::clone() const
{
    LayerGlobalBias* pLayer=new LayerGlobalBias();
	pLayer->weights() = _weight;
    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::init()
{
    _weight.setZero();
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    mOut = mIn.array() + _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalBias::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;

	_gradientWeight(0) = mGradientOut.mean();
	
	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////