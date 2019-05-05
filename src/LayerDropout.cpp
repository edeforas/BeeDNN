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
    _fRate(fRate)
{
    create_mask(iSize);
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
void LayerDropout::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
    if(_bTrainMode)
        mOut = mIn*_mask.asDiagonal(); //in learn mode
    else
        mOut = mIn; // in test mode
}
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta)
{
    mNewDelta= mDelta*_mask.asDiagonal();

    create_mask((int)mInput.cols());
}
///////////////////////////////////////////////////////////////////////////////
void LayerDropout::create_mask(int iSize)
{
    _mask.resize(1, iSize);
	_mask.setConstant(1.f);

    for (int i = 0; i < iSize; i++) //todo distribute a proportion of 1, so we get exactly fRate
    {
        if ( (rand()/(float)RAND_MAX) < _fRate)
            _mask(0, i) = 0.f;
    }

	//inverse dropout as in: https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/)
	//with precise compensation
	_mask*=((float)iSize / _mask.sum());
}
///////////////////////////////////////////////////////////////////////////////
float LayerDropout::get_rate() const
{
    return _fRate;
}
///////////////////////////////////////////////////////////////////////////////
