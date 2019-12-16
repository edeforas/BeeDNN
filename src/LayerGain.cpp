/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGain.h"

///////////////////////////////////////////////////////////////////////////////
LayerGain::LayerGain() :
    Layer(0 , 0, "Gain")
{
    LayerGain::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGain::~LayerGain()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGain::clone() const
{
    LayerGain* pLayer=new LayerGain();
    pLayer->_weight=_weight;
    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGain::init()
{
	_weight.resize(0,0);

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGain::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if(_weight.size()==0)
		_weight.setOnes(1,mIn.cols());
	
    mOut = mIn;

	for (int i = 0; i < mOut.rows(); i++)
		for (int j = 0; j < mOut.cols(); j++)
		{
			mOut(i,j) *= _weight(j);
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerGain::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	//todo manage _bFirstLayer
	_gradientWeight.setZero(1,_weight.cols());
	// compute gradient in and gradient weight
	mGradientIn = mGradientOut;
	for (int i = 0; i < mGradientIn.rows(); i++)
		for (int j = 0; j < mGradientIn.cols(); j++)
		{
			mGradientIn(i, j) *= _weight(j);
			_gradientWeight(j) += mIn(i,j);
		}

	_gradientWeight *= (1.f / mIn.rows());
}
///////////////////////////////////////////////////////////////