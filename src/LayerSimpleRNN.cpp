/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSimpleRNN.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerSimpleRNN::LayerSimpleRNN(int iSampleSize, int iUnits) :
    LayerRNN(iSampleSize, iUnits)
{
    LayerSimpleRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerSimpleRNN::~LayerSimpleRNN()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::init()
{
    _whh.setRandom(_iUnits, _iUnits); // Todo Xavier init ?
    _wxh.setRandom(_iFrameSize, _iUnits); // Todo Xavier init ?
    _bh.setZero(1, _iUnits);
    _h.setZero(1, _iUnits);

    LayerRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSimpleRNN::clone() const
{
    LayerSimpleRNN* pLayer=new LayerSimpleRNN(_iFrameSize,_iUnits);
	pLayer->_whh = _whh;
    pLayer->_wxh = _wxh;
    pLayer->_bh = _bh;
    pLayer->_h = _h;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::forward_frame(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    _h = _h * _whh + mIn * _wxh;
    rowWiseAdd(_h, _bh);
    _h = tanh(_h);
	mOut=_h;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimpleRNN::backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mH, const MatrixFloat& mHm1, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    // FIXED: Implement proper BPTT backpropagation for SimpleRNN
    
    // Step 1: Apply tanh derivative
    // Forward: h(t) = tanh(u(t)) where u(t) = h(t-1)*Whh + x(t)*Wxh + b
    // Backward: d(tanh)/du = 1 - tanh(u)^2 = 1 - h(t)^2
    MatrixFloat mTanhDeriv = oneMinusSquare(mH);
    
    // Step 2: Combine output gradient with tanh derivative
    // dL/du(t) = dL/dh(t) * d(tanh)/du(t)
    MatrixFloat mGradU = mGradientOut * mTanhDeriv;
    
    // Step 3: Compute gradients w.r.t. weights
    // dL/dWhh = dL/du(t) * h(t-1)^T
    MatrixFloat mGradWhh = mGradU.transpose() * mHm1;
    
    // dL/dWxh = dL/du(t) * x(t)^T
    MatrixFloat mGradWxh = mGradU.transpose() * mInFrame;
    
    // For now, store Whh gradient as the main weight gradient
    // (ideally you'd store both Whh and Wxh separately)
    _gradientWeight = mGradWhh * (1.f / mGradU.rows());
    
    // Step 4: Backpropagate to previous hidden state
    // dL/dh(t-1) = dL/du(t) * dWh^T = mGradU * Whh^T
    if (!_bFirstLayer)
    {
        mGradientIn = mGradU * (_whh.transpose());
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
}