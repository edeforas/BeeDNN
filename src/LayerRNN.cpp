/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerRNN.h"
namespace beednn {

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

    if ( (mIn.size() != _iFrameSize) || _bTrainMode)
    {
        // not on-the-fly prediction, reset state on startup
        init();
    }

    MatrixFloat mFrame;   
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
    MatrixFloat mFrame, mGradientOutTemp = mGradientOut, mH, mHm1;
    MatrixFloat mGradientWeightSum;
    Index iNbSamples = mIn.cols() / _iFrameSize;
    
    // BPTT: Process timesteps in REVERSE order (from T-1 down to 0)
    // FIXED: Start from iNbSamples-1 (last timestep), not iNbSamples-2
    // FIXED: Go down to and include iS=0 (first timestep) using >=
    for (Index iS = iNbSamples - 1; iS >= 0; iS--)
    {
        if (iS < (int)_savedH.size())
        {
            mH = _savedH[iS];
            
            // Get previous hidden state (or zero for first timestep)
            if (iS > 0)
                mHm1 = _savedH[iS - 1];
            else
                mHm1.setZero(mH.rows(), mH.cols());  // h(-1) = 0
            
            mFrame = colExtract(mIn, iS * _iFrameSize, iS * _iFrameSize + _iFrameSize);
            
            // Backprop through this timestep
            backpropagation_frame(mFrame, mH, mHm1, mGradientOutTemp, mGradientIn);
            
            // Propagate gradient to previous timestep
            mGradientOutTemp = mGradientIn;
            
            // Accumulate weight gradients
            if (mGradientWeightSum.size() == 0)
                mGradientWeightSum = _gradientWeight;
            else
                mGradientWeightSum += _gradientWeight;
        }
    }
 
    // compute mean of _gradientWeight
    if (iNbSamples > 0 && mGradientWeightSum.size() > 0)
        _gradientWeight = mGradientWeightSum * (1.f / iNbSamples);
    
    // ========== GRADIENT CLIPPING ==========
    // Prevent exploding gradients during backpropagation through time
    // This is critical for RNNs with long sequences
    // Clip weight gradients to prevent instability
    clipGradients(_gradientWeight, 5.0f);
    
    // Also clip input gradients if not at first layer
    // This helps propagate stable gradients to previous layers
    if (!_bFirstLayer)
        clipGradients(mGradientIn, 5.0f);
}
/////////////////////////////////////////////////////////////////////////////////////////////
void LayerRNN::init()
{
    Layer::init();
    _savedH.clear();
}
/////////////////////////////////////////////////////////////////////////////////////////////
}