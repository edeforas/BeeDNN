/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerUniformNoise.h"

///////////////////////////////////////////////////////////////////////////////
LayerUniformNoise::LayerUniformNoise(float fNoise):
    Layer("UniformNoise"),
    _fNoise(fNoise),
	_distUniform(-fNoise, fNoise)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerUniformNoise::~LayerUniformNoise()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerUniformNoise::clone() const
{
    return new LayerUniformNoise(_fNoise);
}
///////////////////////////////////////////////////////////////////////////////
void LayerUniformNoise::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	mOut = mIn;
	if (_bTrainMode && (_fNoise > 0.f) )
	{
		for (Index i = 0; i < mOut.size(); i++)
			mOut(i) += _distUniform(randomEngine());
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerUniformNoise::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;

    (void)mIn;
    mGradientIn= mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
float LayerUniformNoise::get_noise() const
{
    return _fNoise;
}
///////////////////////////////////////////////////////////////////////////////
