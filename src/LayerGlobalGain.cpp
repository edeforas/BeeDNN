/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalGain.h"

///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::LayerGlobalGain() :
    Layer("GlobalGain")
{ }
///////////////////////////////////////////////////////////////////////////////
LayerGlobalGain::~LayerGlobalGain()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalGain::clone() const
{
    LayerGlobalGain* pLayer=new LayerGlobalGain();
	pLayer->weights() = _weight;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalGain::init()
{
    _weight.setOnes(1,1); // by default

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
    _gradientWeight = ((mIn.transpose())*mGradientOut)*(1.f /mIn.rows());

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut * _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
float LayerGlobalGain::gain() const
{
    return _weight(0);
}
///////////////////////////////////////////////////////////////
bool LayerGlobalGain::has_weight() const
{
	return true;
}
///////////////////////////////////////////////////////////////
