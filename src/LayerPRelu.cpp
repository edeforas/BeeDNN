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
void LayerPRelu::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    mMatOut = mMatIn;

	for (int i = 0; i < mMatOut.rows(); i++)
		for (int j = 0; j < mMatOut.cols(); j++)
		{
			if (mMatOut(i,j) < 0.f)
				mMatOut(i,j) *= _weight(j);
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerPRelu::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	//_gradientWeight.setConstant(30);

	// compute gradient in and gradient weight
	mGradientIn = mGradientOut;
	for (int i = 0; i < mGradientIn.rows(); i++)
		for (int j = 0; j < mGradientIn.cols(); j++)
		{
			if (mInput(i, j) < 0.f)
			{
				mGradientIn(i, j) *= _weight(j);
				_gradientWeight(j) += _weight(j);
			}
			else
			{
				_gradientWeight(j) = 0.f;
			}
		}
}
///////////////////////////////////////////////////////////////
