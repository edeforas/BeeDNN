/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerPRelu.h"

///////////////////////////////////////////////////////////////////////////////
LayerPRelu::LayerPRelu() :
    Layer("PRelu")
{
    LayerPRelu::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerPRelu::~LayerPRelu()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPRelu::clone() const
{
    LayerPRelu* pLayer=new LayerPRelu();
    pLayer->_weight=_weight;
	pLayer->_gradientWeight = _gradientWeight;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerPRelu::init()
{
	_weight.resize(0,0);
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerPRelu::predict(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if (_weight.size() == 0)
	{
		_weight.resize(1, mIn.cols());
		_weight.setConstant(0.25f);
		_gradientWeight.resizeLike(_weight);
	}

    mOut = mIn;

	for (Index i = 0; i < mOut.rows(); i++)
		for (Index j = 0; j < mOut.cols(); j++)
		{
			if (mOut(i,j) < 0.f)
				mOut(i,j) *= _weight(j);
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerPRelu::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    _gradientWeight = colWiseMean((mIn.transpose())*mGradientOut);

	if (_bFirstLayer)
		return;

	// compute gradientin
	mGradientIn = mGradientOut;
	for (Index i = 0; i < mGradientIn.rows(); i++)
		for (Index j = 0; j < mGradientIn.cols(); j++)
		{
			if (mIn(i, j) < 0.f)
				mGradientIn(i, j) *= _weight(j);
		}
}
///////////////////////////////////////////////////////////////