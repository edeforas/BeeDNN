/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// GaussianDropout as in: https://keras.io/layers/noise/

#include "LayerGaussianDropout.h"

///////////////////////////////////////////////////////////////////////////////
LayerGaussianDropout::LayerGaussianDropout(float fProba):
    Layer(0,0,"GaussianDropout"),
    _fProba(fProba),
	_fStdev( sqrtf(_fProba / (1.f - _fProba))),
	_distNormal(1.f, _fStdev)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerGaussianDropout::~LayerGaussianDropout()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGaussianDropout::clone() const
{
    return new LayerGaussianDropout(_fProba);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianDropout::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if (_bTrainMode)
	{
		_mask.resize(1, mIn.size());

		for (Index i = 0; i < _mask.size(); i++)
			_mask(0, i) = _distNormal(randomEngine());
		
		mOut = mIn * _mask.asDiagonal();
	}
	else
        mOut = mIn;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGaussianDropout::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;
	assert(_bTrainMode);

	if (_bFirstLayer)
		return;

	mGradientIn= mGradientOut*_mask.asDiagonal();
}
///////////////////////////////////////////////////////////////////////////////
float LayerGaussianDropout::get_proba() const
{
    return _fProba;
}
///////////////////////////////////////////////////////////////////////////////
