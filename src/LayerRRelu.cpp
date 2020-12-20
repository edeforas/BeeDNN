/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerRRelu.h"

// as in : https://arxiv.org/pdf/1505.00853.pdf

///////////////////////////////////////////////////////////////////////////////
LayerRRelu::LayerRRelu(float alpha1, float alpha2) :
    Layer("RRelu")
{
	_alpha1=alpha1;
	_alpha2=alpha2;
	_invAlpha1=1.f/alpha1;
	_invAlpha2=1.f/alpha2;
	_invAlphaMean=(_invAlpha1+_invAlpha2)*0.5f;
	
    LayerRRelu::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerRRelu::~LayerRRelu()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerRRelu::clone() const
{
    return new LayerRRelu(_alpha1,_alpha2);
}
///////////////////////////////////////////////////////////////////////////////
void LayerRRelu::init()
{
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerRRelu::predict(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn;
	if(_bTrainMode)
	{
		_slopes.resize(mIn.rows(), mIn.cols());
		setRandomUniform(_slopes, _invAlpha1, _invAlpha2);

		for (Index i = 0; i < mOut.size(); i++)
			if (mOut(i) < 0.f)
				mOut(i) *= _slopes(i);
	}
	else
	{
		for (Index i = 0; i < mOut.size(); i++)
			if (mOut(i) < 0.f)
				mOut(i) *= _invAlphaMean;
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerRRelu::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;

	// compute gradientin
	mGradientIn = mGradientOut;
	for (Index i = 0; i < mGradientIn.size(); i++)
	{
		if (mIn(i) < 0.f)
			mGradientIn(i) *= _slopes(i);
	}
}
///////////////////////////////////////////////////////////////
void LayerRRelu::get_params(float& alpha1, float& alpha2) const
{
	alpha1 = _alpha1;
	alpha2 = _alpha2;
}
///////////////////////////////////////////////////////////////
