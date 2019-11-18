/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// GaussianDropout as in: https://keras.io/layers/noise/

#include "LayerGaussianDropout.h"

///////////////////////////////////////////////////////////////////////////////
LayerGaussianDropout::LayerGaussianDropout(int iSize,float fProba):
    Layer(iSize,iSize,"GaussianDropout"),
    _fProba(fProba),
	_fStdev( sqrtf(_fProba / (1.f - _fProba))),
	_distNormal(1., _fStdev)
{
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
    _mask.resize(1, _iInSize);

    for (int i = 0; i < _iInSize; i++)
		_mask(0, i) = _distNormal(_RNGgenerator);
}
///////////////////////////////////////////////////////////////////////////////
float LayerGaussianDropout::get_proba() const
{
    return _fProba;
}
///////////////////////////////////////////////////////////////////////////////
