/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerLSTM.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerLSTM::LayerLSTM(int iFrameSize, int iUnits) :
    LayerRNN(iFrameSize, iUnits)
{
    LayerLSTM::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerLSTM::~LayerLSTM()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerLSTM::init()
{
    // Initialize forget gate weights
    _wxf.setRandom(_iFrameSize, _iUnits);
    _whf.setRandom(_iUnits, _iUnits);
    _bf.setZero(1, _iUnits);
    
    // Initialize input gate weights
    _wxi.setRandom(_iFrameSize, _iUnits);
    _whi.setRandom(_iUnits, _iUnits);
    _bi.setZero(1, _iUnits);
    
    // Initialize output gate weights
    _wxo.setRandom(_iFrameSize, _iUnits);
    _who.setRandom(_iUnits, _iUnits);
    _bo.setZero(1, _iUnits);
    
    // Initialize candidate cell state weights
    _wxc.setRandom(_iFrameSize, _iUnits);
    _whc.setRandom(_iUnits, _iUnits);
    _bc.setZero(1, _iUnits);
    
    // Initialize hidden state and cell state
    _h.setZero(1, _iUnits);
    _c.setZero(1, _iUnits);

    LayerRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerLSTM::clone() const
{
    LayerLSTM* pLayer = new LayerLSTM(_iFrameSize, _iUnits);
    
    // Clone forget gate
    pLayer->_wxf = _wxf;
    pLayer->_whf = _whf;
    pLayer->_bf = _bf;
    
    // Clone input gate
    pLayer->_wxi = _wxi;
    pLayer->_whi = _whi;
    pLayer->_bi = _bi;
    
    // Clone output gate
    pLayer->_wxo = _wxo;
    pLayer->_who = _who;
    pLayer->_bo = _bo;
    
    // Clone candidate cell state
    pLayer->_wxc = _wxc;
    pLayer->_whc = _whc;
    pLayer->_bc = _bc;
    
    // Clone states
    pLayer->_h = _h;
    pLayer->_c = _c;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerLSTM::forward_frame(const MatrixFloat& mInFrame, MatrixFloat& mOut)
{
    // Adapt to batch size
    if (_h.rows() != mInFrame.rows())
    {
        _h.setZero(mInFrame.rows(), _iUnits);
        _c.setZero(mInFrame.rows(), _iUnits);
    }
    
    // Save previous cell state for backpropagation
    _saved_c = _c;
    
    // ========== FORGET GATE ==========
    // f(t) = sigmoid(Wxf*x(t) + Whf*h(t-1) + bf)
    MatrixFloat mForgetInput = mInFrame * _wxf + _h * _whf;
    rowWiseAdd(mForgetInput, _bf);
    _forget_gate = mForgetInput;
    
    // Apply sigmoid activation
    for (Index i = 0; i < _forget_gate.size(); i++)
    {
        float x = _forget_gate(i);
        _forget_gate(i) = 1.0f / (1.0f + expf(-x));  // sigmoid
    }
    
    // ========== INPUT GATE ==========
    // i(t) = sigmoid(Wxi*x(t) + Whi*h(t-1) + bi)
    MatrixFloat mInputInput = mInFrame * _wxi + _h * _whi;
    rowWiseAdd(mInputInput, _bi);
    _input_gate = mInputInput;
    
    // Apply sigmoid activation
    for (Index i = 0; i < _input_gate.size(); i++)
    {
        float x = _input_gate(i);
        _input_gate(i) = 1.0f / (1.0f + expf(-x));  // sigmoid
    }
    
    // ========== OUTPUT GATE ==========
    // o(t) = sigmoid(Wxo*x(t) + Who*h(t-1) + bo)
    MatrixFloat mOutputInput = mInFrame * _wxo + _h * _who;
    rowWiseAdd(mOutputInput, _bo);
    _output_gate = mOutputInput;
    
    // Apply sigmoid activation
    for (Index i = 0; i < _output_gate.size(); i++)
    {
        float x = _output_gate(i);
        _output_gate(i) = 1.0f / (1.0f + expf(-x));  // sigmoid
    }
    
    // ========== CANDIDATE CELL STATE ==========
    // c'(t) = tanh(Wxc*x(t) + Whc*h(t-1) + bc)
    MatrixFloat mCandidateInput = mInFrame * _wxc + _h * _whc;
    rowWiseAdd(mCandidateInput, _bc);
    _candidate_c = tanh(mCandidateInput);
    
    // ========== CELL STATE ==========
    // c(t) = f(t) * c(t-1) + i(t) * c'(t)
    MatrixFloat mNewC(_c.rows(), _c.cols());
    for (Index i = 0; i < _c.size(); i++)
    {
        mNewC(i) = _forget_gate(i) * _c(i) + _input_gate(i) * _candidate_c(i);
    }
    _c = mNewC;
    
    // ========== HIDDEN STATE ==========
    // h(t) = o(t) * tanh(c(t))
    MatrixFloat mTanhC = tanh(_c);
    _h = _output_gate;
    for (Index i = 0; i < _h.size(); i++)
    {
        _h(i) = _h(i) * mTanhC(i);
    }
    
    mOut = _h;
}
///////////////////////////////////////////////////////////////////////////////
void LayerLSTM::backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mH, const MatrixFloat& mHm1, 
                                      const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    // LSTM backpropagation through time
    // This is a simplified implementation that computes main gradients
    
    // dL/dh(t) = mGradientOut (from next layer or loss)
    MatrixFloat mGradH = mGradientOut;
    
    // ========== OUTPUT GATE GRADIENT ==========
    // dL/do(t) = dL/dh(t) * tanh(c(t))
    MatrixFloat mTanhC = tanh(_c);
    MatrixFloat mGradOutputGate = mGradH;
    for (Index i = 0; i < mGradOutputGate.size(); i++)
    {
        mGradOutputGate(i) = mGradOutputGate(i) * mTanhC(i);
    }
    
    // Apply sigmoid derivative: d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
    for (Index i = 0; i < mGradOutputGate.size(); i++)
    {
        float o = _output_gate(i);
        mGradOutputGate(i) = mGradOutputGate(i) * o * (1.0f - o);
    }
    
    // ========== CELL STATE GRADIENT ==========
    // dL/dc(t) = dL/dh(t) * o(t) * d(tanh(c(t)))/dc(t)
    MatrixFloat mTanhDeriv = oneMinusSquare(mTanhC);
    MatrixFloat mGradC = mGradH;
    for (Index i = 0; i < mGradC.size(); i++)
    {
        mGradC(i) = mGradC(i) * _output_gate(i) * mTanhDeriv(i);
    }
    
    // ========== INPUT GATE GRADIENT ==========
    // dL/di(t) = dL/dc(t) * c'(t)
    MatrixFloat mGradInputGate = mGradC;
    for (Index i = 0; i < mGradInputGate.size(); i++)
    {
        mGradInputGate(i) = mGradInputGate(i) * _candidate_c(i);
    }
    
    // Apply sigmoid derivative
    for (Index i = 0; i < mGradInputGate.size(); i++)
    {
        float ii = _input_gate(i);
        mGradInputGate(i) = mGradInputGate(i) * ii * (1.0f - ii);
    }
    
    // ========== FORGET GATE GRADIENT ==========
    // dL/df(t) = dL/dc(t) * c(t-1)
    MatrixFloat mGradForgetGate = mGradC;
    for (Index i = 0; i < mGradForgetGate.size(); i++)
    {
        mGradForgetGate(i) = mGradForgetGate(i) * _saved_c(i);
    }
    
    // Apply sigmoid derivative
    for (Index i = 0; i < mGradForgetGate.size(); i++)
    {
        float f = _forget_gate(i);
        mGradForgetGate(i) = mGradForgetGate(i) * f * (1.0f - f);
    }
    
    // ========== CANDIDATE CELL STATE GRADIENT ==========
    // dL/dc'(t) = dL/dc(t) * i(t)
    MatrixFloat mGradCandidateC = mGradC;
    for (Index i = 0; i < mGradCandidateC.size(); i++)
    {
        mGradCandidateC(i) = mGradCandidateC(i) * _input_gate(i);
    }
    
    // Apply tanh derivative: d(tanh)/dx = 1 - tanh(x)^2
    MatrixFloat mCandidateTanhDeriv = oneMinusSquare(_candidate_c);
    mGradCandidateC = mGradCandidateC * mCandidateTanhDeriv;
    
    // ========== WEIGHT GRADIENTS ==========
    // Simplified: accumulate main candidate cell state gradient
    // In a full implementation, you would accumulate gradients for all gates
    _gradientWeight = mGradCandidateC.transpose() * mInFrame;
    _gradientWeight *= (1.f / mGradCandidateC.rows());
    
    // ========== GRADIENT TO PREVIOUS HIDDEN STATE ==========
    if (!_bFirstLayer)
    {
        // dL/dh(t-1) comes from all gates through their weight connections
        mGradientIn = mGradForgetGate * _whf.transpose() +
                      mGradInputGate * _whi.transpose() +
                      mGradOutputGate * _who.transpose() +
                      mGradCandidateC * _whc.transpose();
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
}