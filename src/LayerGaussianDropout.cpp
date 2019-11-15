/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// GaussianDropout as in: https://keras.io/layers/noise/

#include "LayerGaussianDropout.h"
#include <random>

///////////////////////////////////////////////////////////////////////////////
LayerGaussianDropout::LayerGaussianDropout(int iSize,float fProba):
    Layer(iSize,iSize,"GaussianDropout"),
    _fProba(fProba)
{
	if(_fProba>1.f)
		_fProba=1.f;
	
	if(_fProba<0.f)
		_fProba=0.f;
	
	_fStdev=sqrtf(_fProba / (1.f - _fProba));
	
    LayerGaussianDropout::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGaussianDropout::~LayerGaussianDropout()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGaussianDropout::clone() const
{
    return new LayerGaussianDropout(_iInSize,_fProba);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianDropout::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    if(_bTrainMode)
        mOut = mIn*_mask.asDiagonal(); //in learn mode
    else
        mOut = mIn; // in test mode
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianDropout::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;
	assert(_bTrainMode);
    mGradientIn= mGradientOut*_mask.asDiagonal();
	init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianDropout::init()
{
	default_random_engine RNGgenerator; //todo check perfs of init every time
	normal_distribution<float> distNormal(1.f, _fStdev); //todo check perfs of init every time

    _mask.resize(1, _iInSize);

    for (int i = 0; i < _iInSize; i++)
		_mask(0, i) = distNormal(RNGgenerator);
}
///////////////////////////////////////////////////////////////////////////////
float LayerGaussianDropout::get_proba() const
{
    return _fProba;
}
///////////////////////////////////////////////////////////////////////////////
