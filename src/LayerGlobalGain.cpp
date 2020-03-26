/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalGain.h"

///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::LayerGlobalGain():
    Layer( "GlobalGain")
{
    _weight.resize(1,1);
	_gradientWeight.resize(1, 1);
    LayerGlobalGain::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::~LayerGlobalGain()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalGain::clone() const
{
    LayerGlobalGain* pLayer=new LayerGlobalGain();
	pLayer->_weight = _weight;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::init()
{
    _weight.setOnes(); // by default

	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    mOut = mIn * _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    _gradientWeight(0) = ((mIn.transpose())*mGradientOut).mean();

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut * _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
