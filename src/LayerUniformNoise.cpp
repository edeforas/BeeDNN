/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerUniformNoise.h"

#include <random>

///////////////////////////////////////////////////////////////////////////////
LayerUniformNoise::LayerUniformNoise(int iSize,float fNoise):
    Layer(iSize,iSize,"UniformNoise"),
    _fNoise(fNoise)
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
void LayerUniformNoise::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
	if (_bTrainMode && (_fNoise > 0.) )
	{
		default_random_engine RNGgenerator; //todo check perfs of init every time
		std::uniform_real<float> distUniform(-_fNoise, _fNoise); //todo check perfs of init every time

		mOut.resize(mIn.rows(), mIn.cols());

		for (int i = 0; i < mOut.size(); i++)
			mOut(i) = mIn(i) + distUniform(RNGgenerator);
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
