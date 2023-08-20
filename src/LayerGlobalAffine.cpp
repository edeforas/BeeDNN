/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGlobalAffine.h"
namespace bee {

///////////////////////////////////////////////////////////////////////////////
LayerGlobalAffine::LayerGlobalAffine():
    Layer( "GlobalAffine")
{
    _weight.resize(1,1);
	_gradientWeight.resize(1, 1);
    _bias.resize(1,1);
	_gradientBias.resize(1, 1);

	LayerGlobalAffine::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGlobalAffine::~LayerGlobalAffine()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGlobalAffine::clone() const
{
    LayerGlobalAffine* pLayer=new LayerGlobalAffine();
	pLayer->_weight = _weight;
	pLayer->_bias = _bias;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalAffine::init()
{
    _weight.setOnes(); // by default
    _bias.setZero(); // by default

	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalAffine::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    mOut = (mIn * _weight(0)).array()+_bias(0);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGlobalAffine::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    _gradientWeight(0) = ((mIn.transpose())*mGradientOut).mean();
	_gradientBias(0) = mGradientOut.mean();

	if (_bFirstLayer)
		return;

    mGradientIn = mGradientOut * _weight(0);
}
///////////////////////////////////////////////////////////////////////////////
}