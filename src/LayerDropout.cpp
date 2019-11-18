/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerDropout.h"

///////////////////////////////////////////////////////////////////////////////
LayerDropout::LayerDropout(int iSize,float fRate):
    Layer(iSize,iSize,"Dropout"),
    _fRate(fRate),
	_distBernoulli(fRate)
{
    LayerDropout::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDropout::~LayerDropout()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerDropout::clone() const
{
    return new LayerDropout(_iInSize,_fRate);
}
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    if(_bTrainMode && (_fRate!=0.f) )
        mOut = mIn*_mask.asDiagonal(); //in train mode
	else
        mOut = mIn; // in test mode
}
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;
	assert(_bTrainMode);
	
	if(_fRate!=0.f)
	{
		mGradientIn= mGradientOut*_mask.asDiagonal();
		init();
	}
	else
		mGradientIn= mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::init()
{
    _mask.setOnes(1, _iInSize);

    if(_fRate==0.f)
        return;

    for (int i = 0; i < _iInSize; i++)
    {
        if (_distBernoulli(_RNGgenerator))
            _mask(0, i) = 0.f;
    }

	//inverse dropout as in: https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/)
	_mask*=1.f/(1.f-_fRate);
}
///////////////////////////////////////////////////////////////////////////////
float LayerDropout::get_rate() const
{
    return _fRate;
}
///////////////////////////////////////////////////////////////////////////////
