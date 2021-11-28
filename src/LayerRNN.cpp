/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerRNN.h"

///////////////////////////////////////////////////////////////////////////////
LayerRNN::LayerRNN(int iSampleSize, int iUnits) :
    Layer("RNN"),
    _iSampleSize(iSampleSize),
    _iUnits(iUnits)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerRNN::~LayerRNN()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    assert( (mIn.cols() % _iSampleSize)==0); // all samples are concatened horizontaly

    Index iNbSamples = mIn.rows();

    if (mIn.size() != _iSampleSize) // todo use train mode
    {
        // not on-the-fly prediction, reset state on startup
        init();
    }

    MatrixFloat mFrame;

    if (_bTrainMode)
    {
        _savedH.clear();
        _savedH.push_back(_h);
    }
    
    for (Index iS = 0; iS < mIn.cols() - _iSampleSize; iS += _iSampleSize)
    {
        mFrame = colExtract(mIn, iS , iS + _iSampleSize);
	    forward_frame(mFrame,mOut);

        if (_bTrainMode)
            _savedH.push_back(_h);
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    MatrixFloat mFrame,mGradientOutTemp= mGradientOut,mH,mHm1;
    for (Index iS = mIn.cols()-1- _iSampleSize; iS >=0; iS -= _iSampleSize)
    {
        mH = _savedH[iS / _iSampleSize];
        mHm1 = _savedH[iS / _iSampleSize-1];
        mFrame = colExtract(mIn, iS, iS + _iSampleSize);
        backpropagation_frame(mFrame, mH,mHm1,mGradientOutTemp, mGradientIn);
        mGradientOutTemp = mGradientIn;
    }

    // compute mean of _whh
    // 
    // compute mean of _gradientWhh
    // optimize
}
/////////////////////////////////////////////////////////////////////////////////////////////
void LayerRNN::init()
{
    Layer::init();
}
/////////////////////////////////////////////////////////////////////////////////////////////