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

// GRU (Gated Recurrent Unit) implementation
// Based on: https://arxiv.org/abs/1406.1078
// Equations:
//   r(t) = sigmoid(Wxr*x(t) + Whr*h(t-1) + br)  // reset gate
//   z(t) = sigmoid(Wxz*x(t) + Whz*h(t-1) + bz)  // update gate
//   h'(t) = tanh(Wxh*x(t) + Wh*(r(t) * h(t-1)) + bh)  // candidate hidden state
//   h(t) = (1-z(t)) * h'(t) + z(t) * h(t-1)     // final hidden state

namespace beednn {
class LayerGRU : public LayerRNN
{
public:
    explicit LayerGRU(int iFrameSize, int iUnits);
    virtual ~LayerGRU();
    virtual void init() override;

    virtual Layer* clone() const override;
    virtual void forward_frame(const MatrixFloat& mInFrame, MatrixFloat& mOut) override;
    virtual void backpropagation_frame(const MatrixFloat& mInFrame, const MatrixFloat& mH, const MatrixFloat& mHm1, 
                                       const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

private:
    // Weight matrices
    MatrixFloat _wxr, _whr, _br;  // Reset gate weights
    MatrixFloat _wxz, _whz, _bz;  // Update gate weights
    MatrixFloat _wxh, _whh, _bh;  // Candidate hidden state weights
    
    // For backpropagation - save intermediate values
    MatrixFloat _reset_gate, _update_gate, _candidate_h;
};
}