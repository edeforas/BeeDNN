/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// TERELU as in : https://arxiv.org/ftp/arxiv/papers/2006/2006.02797.pdf
// 

#include "LayerTERELU.h"

///////////////////////////////////////////////////////////////////////////////
LayerTERELU::LayerTERELU() :
    Layer("TERELU")
{
    LayerTERELU::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerTERELU::~LayerTERELU()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerTERELU::clone() const
{
    LayerTERELU* pLayer=new LayerTERELU();
    pLayer->_weight=_weight;
	pLayer->_gradientWeight = _gradientWeight;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerTERELU::init()
{
	_weight.resize(0,0);
	_alpha = 1.f;
	_mu = 1.f;
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerTERELU::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if (_weight.size() == 0)
	{
		_weight.setConstant(1, mIn.cols(), 1.f); // beta initialized as 1.f
		_gradientWeight.resizeLike(_weight);
	}

    mOut = mIn;

	for (Index i = 0; i < mOut.rows(); i++)
		for (Index j = 0; j < mOut.cols(); j++)
		{
			if (mIn(i,j) <= 0.f)
				mOut(i,j) = _alpha*expm1f(mIn(i, j));
			else if (mIn(i, j)>=_mu)
				mOut(i, j) = _weight(j) * (_mu- expm1f(_mu-mIn(i, j)));
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerTERELU::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	_gradientWeight.setZero();
	
	// compute loss gradient vs weight
	for (Index i = 0; i < mIn.rows(); i++)
		for (Index j = 0; j < mIn.cols(); j++)
		{
			if (mIn(i, j) >= _mu)
				_gradientWeight(0,j) += (_mu - expm1f(_mu - mIn(i, j)));
		}

	_gradientWeight/=(float)mIn.rows();
	
	if (_bFirstLayer)
		return;

	// compute loss gradient vs input
	mGradientIn = mGradientOut;
	for (Index i = 0; i < mGradientIn.rows(); i++)
		for (Index j = 0; j < mGradientIn.cols(); j++)
		{
			if (mIn(i, j) <= 0.f)
				mGradientIn(i, j) *= _alpha * expf(mIn(i, j));
			else if (mIn(i, j) >= _mu)
				mGradientIn(i, j) *= _weight(j) * expf(_mu - mIn(i, j));
		}
}
///////////////////////////////////////////////////////////////