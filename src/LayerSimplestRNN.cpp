/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSimplestRNN.h"
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerSimplestRNN::LayerSimplestRNN(int iFrameSize) :
    LayerRNN(iFrameSize, iFrameSize)
{
    LayerSimplestRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerSimplestRNN::~LayerSimplestRNN()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerSimplestRNN::init()
{
    _weight.setRandom(_iUnits, _iUnits); // Todo Xavier init ?

    LayerRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSimplestRNN::clone() const
{
    LayerSimplestRNN* pLayer=new LayerSimplestRNN(_iFrameSize);
	pLayer->_weight = _weight;
    pLayer->_h = _h;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimplestRNN::forward_frame(const MatrixFloat& mInFrame, MatrixFloat& mOut)
{
    if (_h.rows() != mInFrame.rows())  // adapt to batch size
        _h.setZero(mInFrame.rows(), _iUnits);

    MatrixFloat u = _h * _weight + mInFrame;
    _h = u;// tanh(u);
	mOut=_h;
}
///////////////////////////////////////////////////////////////////////////////
void LayerSimplestRNN::backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mH, const MatrixFloat& mHm1, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    // FIXED: Use the input gradient mGradientOut, not mH!
    // mGradientOut is dL/dy(t) - the gradient from the loss
    // mH is h(t) - the hidden state value
    
    // Since we're using linear activation (no tanh applied in forward):
    // d(output)/d(input) = 1
    // So gradient through this layer = mGradientOut * 1
    
    MatrixFloat mGradU = mGradientOut;

    // Compute weight gradient:
    // dL/dW = dL/dU * dU/dW = mGradU * h(t-1)^T
    _gradientWeight = mGradU.transpose() * mHm1;
    _gradientWeight *= (1.f / mGradU.rows());

    if (!_bFirstLayer)
    {
        // Backpropagate to previous hidden state:
        // dL/dh(t-1) = dL/dU * dU/dh(t-1) = mGradU * W^T
        mGradientIn = mGradU * (_weight.transpose());
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
}