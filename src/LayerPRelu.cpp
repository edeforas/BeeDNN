/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerPRelu.h"

///////////////////////////////////////////////////////////////////////////////
LayerPRelu::LayerPRelu(int iInSize) :
    Layer(iInSize , iInSize, "PRelu")
{
    LayerPRelu::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerPRelu::~LayerPRelu()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPRelu::clone() const
{
    LayerPRelu* pLayer=new LayerPRelu(_iInSize);
    pLayer->_weight=_weight;
    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerPRelu::init()
{
	assert(_iInSize > 0);
	assert(_iOutSize > 0);
	
	_weight.resize(1,_iInSize);
	_weight.setConstant(0.25f);

	_gradientWeight.resizeLike(_weight);

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerPRelu::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    mOut = mIn;

	for (int i = 0; i < mOut.rows(); i++)
		for (int j = 0; j < mOut.cols(); j++)
		{
			if (mOut(i,j) < 0.f)
				mOut(i,j) *= _weight(j);
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerPRelu::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	//todo manage _bFirstLayer

	// compute gradient in and gradient weight
	mGradientIn = mGradientOut;
	for (int i = 0; i < mGradientIn.rows(); i++)
		for (int j = 0; j < mGradientIn.cols(); j++)
		{
			if (mIn(i, j) < 0.f)
			{
				mGradientIn(i, j) *= _weight(j);
				_gradientWeight(j) += _weight(j);
			}
			else
			{
				_gradientWeight(j) = 0.f;
			}
		}

	_gradientWeight *= (1.f / mIn.rows());
}
///////////////////////////////////////////////////////////////
bool LayerPRelu::has_weight() const
{
	return true;
}
///////////////////////////////////////////////////////////////
