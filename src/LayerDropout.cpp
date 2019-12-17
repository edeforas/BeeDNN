/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

//inverse dropout as in: https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/

#include "LayerDropout.h"

///////////////////////////////////////////////////////////////////////////////
LayerDropout::LayerDropout(float fRate):
    Layer(0,0,"Dropout"),
    _fRate(fRate),
	_distBernoulli(fRate)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerDropout::~LayerDropout()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerDropout::clone() const
{
    return new LayerDropout(_fRate);
}
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if (_bTrainMode && (_fRate != 0.f))
	{
		_mask.setConstant(mIn.rows(), mIn.cols(), 1.f / (1.f - _fRate));
		
		for(int i=0;i< _mask.size();i++)
			if (_distBernoulli(randomEngine()))
				_mask(i) = 0.f;

		mOut = mIn.cwiseProduct(_mask);
	}
	else
        mOut = mIn;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	if (_bFirstLayer)
		return;

	if(_fRate!=0.f)
		mGradientIn= mGradientOut.cwiseProduct(_mask);
	else
		mGradientIn= mGradientOut;
}
///////////////////////////////////////////////////////////////////////////////
float LayerDropout::get_rate() const
{
    return _fRate;
}
///////////////////////////////////////////////////////////////////////////////
