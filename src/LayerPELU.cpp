/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// PELU as in : https://arxiv.org/pdf/1605.09332.pdf

#include "LayerPELU.h"

///////////////////////////////////////////////////////////////////////////////
LayerPELU::LayerPELU() :
    Layer("PELU")
{
    LayerPELU::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerPELU::~LayerPELU()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerPELU::clone() const
{
    LayerPELU* pLayer=new LayerPELU();
    pLayer->_weight=_weight;
	pLayer->_gradientWeight = _gradientWeight;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerPELU::init()
{
	_weight.resize(0,0);
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerPELU::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if (_weight.size() == 0)
	{
		_weight.resize(2, mIn.cols()); // 2 parameters a and b
		_weight.setConstant(1.f);
		_gradientWeight.resizeLike(_weight);
	}

    mOut = mIn;

	for (Index i = 0; i < mOut.rows(); i++)
		for (Index j = 0; j < mOut.cols(); j++)
		{
			if (mOut(i,j) > 0.f)
				mOut(i,j) *= _weight(0,j)/_weight(1,j); f(h)=h*a/b
			else
				mOut(i,j) = _weight(0,j)*(expm1f(mOut(i,j)/_weight(1,j)); // f(h)=a*(exp(h/b)-1)
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerPELU::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	TODODODODOD
	
	_gradientWeight.setZero();
	
	// compute weight gradient
	for (Index i = 0; i < mIn.rows(); i++)
		for (Index j = 0; j < mIn.cols(); j++)
		{
			if (mIn(i, j) < 0.f)
				_gradientWeight(0,j) += mIn(i, j)*mGradientOut(0,j);
		}

	_gradientWeight/=(float)mIn.rows();
	
	if (_bFirstLayer)
		return;

	// compute input gradient
	mGradientIn = mGradientOut;
	for (Index i = 0; i < mGradientIn.rows(); i++)
		for (Index j = 0; j < mGradientIn.cols(); j++)
		{
			if (mIn(i, j) < 0.f)
				mGradientIn(i, j) *= _weight(j);
		}
}
///////////////////////////////////////////////////////////////
