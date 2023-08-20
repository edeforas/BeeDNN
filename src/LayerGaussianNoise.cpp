/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGaussianNoise.h"

#include <random>
namespace bee{

///////////////////////////////////////////////////////////////////////////////
LayerGaussianNoise::LayerGaussianNoise(float fNoise):
    Layer("GaussianNoise"),
    _fNoise(fNoise),
	_distNormal(0.f, fNoise)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerGaussianNoise::~LayerGaussianNoise()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGaussianNoise::clone() const
{
    return new LayerGaussianNoise(_fNoise);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianNoise::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn;
	if (_bTrainMode && (_fNoise > 0.f) )
	{
		for (Index i = 0; i < mOut.size(); i++)
			mOut(i) += _distNormal(randomEngine());
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianNoise::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;
 
	if (_bFirstLayer)
		return;

	mGradientIn= mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
float LayerGaussianNoise::get_noise() const
{
    return _fNoise;
}
///////////////////////////////////////////////////////////////////////////////
}