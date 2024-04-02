/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"

// Batch Normalization Layer
// Reference: https://arxiv.org/abs/1502.03167
// 
// Forward pass:
//   1. Compute batch statistics: mean and variance
//   2. Normalize: (x - mean) / sqrt(variance + epsilon)
//   3. Scale and shift: y = gamma * x_norm + beta
//
// During inference, use running statistics (exponential moving average)
// of training batch mean and variance

namespace beednn {
class LayerBatchNormalization : public Layer
{
public:
    explicit LayerBatchNormalization(Index iSize);
    virtual ~LayerBatchNormalization();
    virtual void init() override;

    virtual Layer* clone() const override;
    virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
    virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

private:
    Index _iSize;  // Size of input features
    
    // Learnable parameters
    MatrixFloat _gamma;  // Scale parameter (weight)
    MatrixFloat _beta;   // Shift parameter (bias)
    
    // Gradients for learnable parameters
    MatrixFloat _gradientGamma;
    MatrixFloat _gradientBeta;
    
    // Running statistics for inference (exponential moving average)
    MatrixFloat _runningMean;
    MatrixFloat _runningVariance;
    
    // Training statistics
    MatrixFloat _batchMean;
    MatrixFloat _batchVariance;
    MatrixFloat _normalized;  // Normalized values (x_norm)
    
    // Hyperparameters
    float _fMomentum;      // For exponential moving average (default: 0.99)
    float _fEpsilon;       // Small constant to avoid division by zero (default: 1e-5)
    
public:
    // Setters for hyperparameters
    void set_momentum(float fMomentum);
    void set_epsilon(float fEpsilon);
    
    float get_momentum() const;
    float get_epsilon() const;
};
}