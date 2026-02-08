/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerGRU.h"
#include "Activations.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerGRU::LayerGRU(int iFrameSize, int iUnits) :
    LayerRNN(iFrameSize, iUnits)
{
    LayerGRU::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGRU::~LayerGRU()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerGRU::init()
{
    // Initialize reset gate weights (Wxr, Whr, br)
    _wxr.setRandom(_iFrameSize, _iUnits);
    _whr.setRandom(_iUnits, _iUnits);
    _br.setZero(1, _iUnits);
    
    // Initialize update gate weights (Wxz, Whz, bz)
    _wxz.setRandom(_iFrameSize, _iUnits);
    _whz.setRandom(_iUnits, _iUnits);
    _bz.setZero(1, _iUnits);
    
    // Initialize candidate hidden state weights (Wxh, Whh, bh)
    _wxh.setRandom(_iFrameSize, _iUnits);
    _whh.setRandom(_iUnits, _iUnits);
    _bh.setZero(1, _iUnits);
    
    // Initialize hidden state
    _h.setZero(1, _iUnits);

    LayerRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGRU::clone() const
{
    LayerGRU* pLayer = new LayerGRU(_iFrameSize, _iUnits);
    pLayer->_wxr = _wxr;
    pLayer->_whr = _whr;
    pLayer->_br = _br;
    
    pLayer->_wxz = _wxz;
    pLayer->_whz = _whz;
    pLayer->_bz = _bz;
    
    pLayer->_wxh = _wxh;
    pLayer->_whh = _whh;
    pLayer->_bh = _bh;
    
    pLayer->_h = _h;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGRU::forward_frame(const MatrixFloat& mInFrame, MatrixFloat& mOut)
{
    // Adapt to batch size
    if (_h.rows() != mInFrame.rows())
        _h.setZero(mInFrame.rows(), _iUnits);
    
    // Compute reset gate: r(t) = sigmoid(Wxr*x(t) + Whr*h(t-1) + br)
    MatrixFloat mResetInput = mInFrame * _wxr + _h * _whr;
    rowWiseAdd(mResetInput, _br);
    _reset_gate = mResetInput; // Will apply sigmoid in backprop, for now just store
    
    // Apply sigmoid activation
    for (Index i = 0; i < _reset_gate.size(); i++)
    {
        float x = _reset_gate(i);
        _reset_gate(i) = 1.0f / (1.0f + expf(-x));  // sigmoid
    }
    
    // Compute update gate: z(t) = sigmoid(Wxz*x(t) + Whz*h(t-1) + bz)
    MatrixFloat mUpdateInput = mInFrame * _wxz + _h * _whz;
    rowWiseAdd(mUpdateInput, _bz);
    _update_gate = mUpdateInput;
    
    // Apply sigmoid activation
    for (Index i = 0; i < _update_gate.size(); i++)
    {
        float x = _update_gate(i);
        _update_gate(i) = 1.0f / (1.0f + expf(-x));  // sigmoid
    }
    
    // Compute candidate hidden state: h'(t) = tanh(Wxh*x(t) + Wh*(r(t) * h(t-1)) + bh)
    MatrixFloat mResetH = _reset_gate;
    for (Index i = 0; i < mResetH.size(); i++)
        mResetH(i) = mResetH(i) * _h(i);  // element-wise multiplication
    
    _candidate_h = mInFrame * _wxh + mResetH * _whh;
    rowWiseAdd(_candidate_h, _bh);
    
    // Apply tanh activation
    _candidate_h = tanh(_candidate_h);
    
    // Compute final hidden state: h(t) = (1-z(t)) * h'(t) + z(t) * h(t-1)
    MatrixFloat mNewH = _candidate_h;
    MatrixFloat mOldH = _h;
    
    // h(t) = (1 - z(t)) * h'(t) + z(t) * h(t-1)
    for (Index i = 0; i < _h.size(); i++)
    {
        float z = _update_gate(i);
        _h(i) = (1.0f - z) * _candidate_h(i) + z * _h(i);
    }
    
    mOut = _h;
}
///////////////////////////////////////////////////////////////////////////////
void LayerGRU::backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mH, const MatrixFloat& mHm1, 
                                      const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    // GRU backpropagation is complex due to gating mechanisms
    // We'll implement a simplified version that accumulates gradients
    
    // dL/dh(t) = mGradientOut (from next layer or loss)
    MatrixFloat mGradH = mGradientOut;
    
    // For simplicity, we'll compute gradients for the candidate state
    // dL/dh'(t) = dL/dh(t) * (1 - z(t))
    MatrixFloat mGradCandidateH = mGradH;
    for (Index i = 0; i < mGradCandidateH.size(); i++)
    {
        float z = _update_gate(i);
        mGradCandidateH(i) = mGradCandidateH(i) * (1.0f - z);
    }
    
    // Apply tanh derivative: d(tanh)/dx = 1 - tanh(x)^2
    MatrixFloat mTanhDeriv = oneMinusSquare(_candidate_h);
    mGradCandidateH = mGradCandidateH * mTanhDeriv;
    
    // Compute weight gradients for candidate hidden state
    // dL/dWxh = dL/dh'(t) * x(t)^T
    _gradientWeight = mGradCandidateH.transpose() * mInFrame;
    _gradientWeight *= (1.f / mGradCandidateH.rows());
    
    // Backpropagate to previous hidden state
    // dL/dh(t-1) includes contributions from all three gates
    if (!_bFirstLayer)
    {
        // From candidate state
        mGradientIn = mGradCandidateH * (_whh.transpose());
        
        // Add contribution from reset gate and update gate
        // dL/dh(t-1) += dL/dh(t) * z(t)
        for (Index i = 0; i < mGradientIn.size(); i++)
        {
            float z = _update_gate(i);
            mGradientIn(i) += mGradH(i) * z;
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
}