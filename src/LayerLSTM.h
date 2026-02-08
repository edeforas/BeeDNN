/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"
#include "LayerRNN.h"

// LSTM (Long Short-Term Memory) implementation
// Based on: https://arxiv.org/abs/1406.1074
// Equations:
//   f(t) = sigmoid(Wxf*x(t) + Whf*h(t-1) + bf)      // forget gate
//   i(t) = sigmoid(Wxi*x(t) + Whi*h(t-1) + bi)      // input gate
//   o(t) = sigmoid(Wxo*x(t) + Who*h(t-1) + bo)      // output gate
//   c'(t) = tanh(Wxc*x(t) + Whc*h(t-1) + bc)        // candidate cell state
//   c(t) = f(t) * c(t-1) + i(t) * c'(t)             // cell state
//   h(t) = o(t) * tanh(c(t))                        // hidden state

namespace beednn {
class LayerLSTM : public LayerRNN
{
public:
    explicit LayerLSTM(int iFrameSize, int iUnits);
    virtual ~LayerLSTM();
    virtual void init() override;

    virtual Layer* clone() const override;
    virtual void forward_frame(const MatrixFloat& mInFrame, MatrixFloat& mOut) override;
    virtual void backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mH, const MatrixFloat& mHm1, 
                                       const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

private:
    // Forget gate weights: W_xf, W_hf, b_f
    MatrixFloat _wxf, _whf, _bf;
    
    // Input gate weights: W_xi, W_hi, b_i
    MatrixFloat _wxi, _whi, _bi;
    
    // Output gate weights: W_xo, W_ho, b_o
    MatrixFloat _wxo, _who, _bo;
    
    // Candidate cell state weights: W_xc, W_hc, b_c
    MatrixFloat _wxc, _whc, _bc;
    
    // Cell state (memory)
    MatrixFloat _c;
    
    // For backpropagation - save intermediate values
    MatrixFloat _forget_gate, _input_gate, _output_gate, _candidate_c;
    MatrixFloat _saved_c;  // Previous cell state
};
}