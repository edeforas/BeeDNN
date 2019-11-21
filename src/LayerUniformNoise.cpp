/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerUniformNoise.h"

///////////////////////////////////////////////////////////////////////////////
LayerUniformNoise::LayerUniformNoise(int iSize,float fNoise):
    Layer(iSize,iSize,"UniformNoise"),
    _fNoise(fNoise),
	_distUniform(-fNoise, fNoise)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerUniformNoise::~LayerUniformNoise()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerUniformNoise::clone() const
{
    return new LayerUniformNoise(_iInSize,_fNoise);
}
///////////////////////////////////////////////////////////////////////////////
void LayerUniformNoise::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if (_bTrainMode && (_fNoise > 0.) )
	{
		mOut.resize(mIn.rows(), mIn.cols());

		for (int i = 0; i < mOut.size(); i++)
			mOut(i) = mIn(i) + _distUniform(_RNGgenerator);
	}
	else
		mOut = mIn; // in test mode or sigma==0.
}
///////////////////////////////////////////////////////////////////////////////
void LayerUniformNoise::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;
    mGradientIn= mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
float LayerUniformNoise::get_noise() const
{
    return _fNoise;
}
///////////////////////////////////////////////////////////////////////////////