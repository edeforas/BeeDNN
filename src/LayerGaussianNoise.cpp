/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGaussianNoise.h"

#include <random>

///////////////////////////////////////////////////////////////////////////////
LayerGaussianNoise::LayerGaussianNoise(int iSize,float fStd):
    Layer(iSize,iSize,"GaussianNoise"),
    _fStd(fStd)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerGaussianNoise::~LayerGaussianNoise()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGaussianNoise::clone() const
{
    return new LayerGaussianNoise(_iInSize,_fStd);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianNoise::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
	if (_bTrainMode && (_fStd > 0.) )
	{
		default_random_engine RNGgenerator; //todo check perfs of init every time
		normal_distribution<float> distNormal(0.f, _fStd); //todo check perfs of init every time

		mOut.resize(mIn.rows(), mIn.cols());

		for (int i = 0; i < mOut.size(); i++)
			mOut(i) = mIn(i) + distNormal(RNGgenerator);
	}
	else
		mOut = mIn; // in test mode or sigma==0.
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianNoise::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta)
{
    (void)mInput;
    mNewDelta= mDelta;
}
///////////////////////////////////////////////////////////////////////////////
float LayerGaussianNoise::get_std() const
{
    return _fStd;
}
///////////////////////////////////////////////////////////////////////////////
