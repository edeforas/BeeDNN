/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerRNN.h"

///////////////////////////////////////////////////////////////////////////////
LayerRNN::LayerRNN(int iFrameSize, int iUnits) :
    Layer("RNN"),
    _iFrameSize(iFrameSize),
    _iUnits(iUnits)
{ }
///////////////////////////////////////////////////////////////////////////////
LayerRNN::~LayerRNN()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    assert( (mIn.cols() % _iFrameSize)==0); // all samples are concatened horizontaly

    if (mIn.size() != _iFrameSize) // todo use train mode
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
    
    for (Index iS = 0; iS < mIn.cols() - _iFrameSize; iS += _iFrameSize)
    {
        mFrame = colExtract(mIn, iS , iS + _iFrameSize);
	    forward_frame(mFrame,mOut);

        if (_bTrainMode)
            _savedH.push_back(_h);
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    MatrixFloat mFrame,mGradientOutTemp= mGradientOut,mH,mHm1;
    MatrixFloat mGradientWeightSum;
    Index iNbSamples = mIn.cols() / _iFrameSize;
    for (Index iS = iNbSamples-1; iS >0; iS --)
    {
        mH = _savedH[iS];
        mHm1 = _savedH[iS-1];
        mFrame = colExtract(mIn, iS* _iFrameSize, iS* _iFrameSize + _iFrameSize);
        backpropagation_frame(mFrame, mH,mHm1,mGradientOutTemp, mGradientIn);
        mGradientOutTemp = mGradientIn;

        //sum gradient weights
        if (mGradientWeightSum.size() == 0)
            mGradientWeightSum = _gradientWeight;
        else
            mGradientWeightSum += _gradientWeight;
    }
 
    // compute mean of _gradientWeight
    _gradientWeight = mGradientWeightSum * (1.f / iNbSamples);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void LayerRNN::init()
{
    Layer::init();
}
/////////////////////////////////////////////////////////////////////////////////////////////