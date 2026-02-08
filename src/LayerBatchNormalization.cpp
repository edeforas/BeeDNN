/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerBatchNormalization.h"
#include <cmath>

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerBatchNormalization::LayerBatchNormalization(Index iSize) :
    Layer("BatchNormalization"),
    _iSize(iSize),
    _fMomentum(0.99f),
    _fEpsilon(1e-5f)
{
    LayerBatchNormalization::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerBatchNormalization::~LayerBatchNormalization()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerBatchNormalization::init()
{
    // Initialize learnable parameters
    _gamma.setOnes(1, _iSize);      // Scale initialized to 1
    _beta.setZero(1, _iSize);       // Shift initialized to 0
    
    // Initialize gradients
    _gradientGamma.setZero(1, _iSize);
    _gradientBeta.setZero(1, _iSize);
    
    // Initialize running statistics (for inference)
    _runningMean.setZero(1, _iSize);
    _runningVariance.setOnes(1, _iSize);
    
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerBatchNormalization::clone() const
{
    LayerBatchNormalization* pLayer = new LayerBatchNormalization(_iSize);
    pLayer->_gamma = _gamma;
    pLayer->_beta = _beta;
    pLayer->_runningMean = _runningMean;
    pLayer->_runningVariance = _runningVariance;
    pLayer->_fMomentum = _fMomentum;
    pLayer->_fEpsilon = _fEpsilon;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerBatchNormalization::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    assert(mIn.cols() == _iSize);
    
    Index iBatchSize = mIn.rows();
    
    if (_bTrainMode)
    {
        // ========== TRAINING MODE ==========
        // Compute batch mean and variance
        
        // Mean: mean = sum(x) / N
        _batchMean = colWiseMean(mIn);
        
        // Variance: var = sum((x - mean)^2) / N
        MatrixFloat mCentered = mIn;
        for (Index i = 0; i < mCentered.rows(); i++)
        {
            for (Index j = 0; j < mCentered.cols(); j++)
            {
                mCentered(i, j) = mCentered(i, j) - _batchMean(j);
            }
        }
        
        // Compute variance (element-wise)
        _batchVariance.setZero(1, _iSize);
        for (Index i = 0; i < mCentered.rows(); i++)
        {
            for (Index j = 0; j < mCentered.cols(); j++)
            {
                _batchVariance(j) += mCentered(i, j) * mCentered(i, j);
            }
        }
        _batchVariance = _batchVariance * (1.0f / iBatchSize);
        
        // Normalize: x_norm = (x - mean) / sqrt(var + eps)
        _normalized = mCentered;
        for (Index i = 0; i < _normalized.rows(); i++)
        {
            for (Index j = 0; j < _normalized.cols(); j++)
            {
                float fStd = sqrtf(_batchVariance(j) + _fEpsilon);
                _normalized(i, j) = _normalized(i, j) / fStd;
            }
        }
        
        // Update running statistics using exponential moving average
        // running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        for (Index j = 0; j < _iSize; j++)
        {
            _runningMean(j) = _fMomentum * _runningMean(j) + (1.0f - _fMomentum) * _batchMean(j);
            _runningVariance(j) = _fMomentum * _runningVariance(j) + (1.0f - _fMomentum) * _batchVariance(j);
        }
    }
    else
    {
        // ========== INFERENCE MODE ==========
        // Use running statistics instead of batch statistics
        
        MatrixFloat mCentered = mIn;
        for (Index i = 0; i < mCentered.rows(); i++)
        {
            for (Index j = 0; j < mCentered.cols(); j++)
            {
                mCentered(i, j) = mCentered(i, j) - _runningMean(j);
            }
        }
        
        _normalized = mCentered;
        for (Index i = 0; i < _normalized.rows(); i++)
        {
            for (Index j = 0; j < _normalized.cols(); j++)
            {
                float fStd = sqrtf(_runningVariance(j) + _fEpsilon);
                _normalized(i, j) = _normalized(i, j) / fStd;
            }
        }
    }
    
    // Scale and shift: y = gamma * x_norm + beta
    mOut.resizeLike(mIn);
    for (Index i = 0; i < mOut.rows(); i++)
    {
        for (Index j = 0; j < mOut.cols(); j++)
        {
            mOut(i, j) = _gamma(j) * _normalized(i, j) + _beta(j);
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerBatchNormalization::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    assert(mIn.cols() == _iSize);
    assert(mGradientOut.cols() == _iSize);
    
    Index iBatchSize = mIn.rows();
    
    // ========== GRADIENT w.r.t. GAMMA AND BETA ==========
    // dL/dgamma = sum(dL/dy * x_norm)
    // dL/dbeta = sum(dL/dy)
    
    _gradientGamma.setZero(1, _iSize);
    _gradientBeta.setZero(1, _iSize);
    
    for (Index i = 0; i < mGradientOut.rows(); i++)
    {
        for (Index j = 0; j < mGradientOut.cols(); j++)
        {
            _gradientGamma(j) += mGradientOut(i, j) * _normalized(i, j);
            _gradientBeta(j) += mGradientOut(i, j);
        }
    }
    
    // Average over batch
    _gradientGamma = _gradientGamma * (1.0f / iBatchSize);
    _gradientBeta = _gradientBeta * (1.0f / iBatchSize);
    
    // ========== GRADIENT w.r.t. INPUT ==========
    // This is more complex, but here's a simplified version
    // For full derivation, see https://arxiv.org/abs/1502.03167
    
    // dL/dx_norm = dL/dy * gamma
    MatrixFloat mGradNormalized = mGradientOut;
    for (Index i = 0; i < mGradNormalized.rows(); i++)
    {
        for (Index j = 0; j < mGradNormalized.cols(); j++)
        {
            mGradNormalized(i, j) = mGradNormalized(i, j) * _gamma(j);
        }
    }
    
    // Compute variance gradient
    // dL/dvar = sum(dL/dx_norm * (x - mean) * -0.5 * (var + eps)^-1.5)
    MatrixFloat mGradVariance(1, _iSize);
    mGradVariance.setZero(1, _iSize);
    
    MatrixFloat mCentered = mIn;
    for (Index i = 0; i < mCentered.rows(); i++)
    {
        for (Index j = 0; j < mCentered.cols(); j++)
        {
            mCentered(i, j) = mCentered(i, j) - _batchMean(j);
        }
    }
    
    for (Index i = 0; i < mGradNormalized.rows(); i++)
    {
        for (Index j = 0; j < mGradNormalized.cols(); j++)
        {
            float fStd = sqrtf(_batchVariance(j) + _fEpsilon);
            mGradVariance(j) += mGradNormalized(i, j) * mCentered(i, j) * 
                                (-0.5f) / (fStd * fStd * fStd);
        }
    }
    
    // Compute mean gradient
    // dL/dmean = sum(dL/dx_norm * -1 / sqrt(var + eps)) + dL/dvar * sum(-2 * (x - mean)) / N
    MatrixFloat mGradMean(1, _iSize);
    mGradMean.setZero(1, _iSize);
    
    for (Index i = 0; i < mGradNormalized.rows(); i++)
    {
        for (Index j = 0; j < mGradNormalized.cols(); j++)
        {
            float fStd = sqrtf(_batchVariance(j) + _fEpsilon);
            mGradMean(j) += mGradNormalized(i, j) * (-1.0f / fStd);
            mGradMean(j) += mGradVariance(j) * mCentered(i, j) * (-2.0f / iBatchSize);
        }
    }
    
    // Compute input gradient
    // dL/dx = dL/dx_norm * (1 / sqrt(var + eps)) + dL/dvar * (2 * (x - mean) / N) + dL/dmean * (1 / N)
    mGradientIn.resizeLike(mIn);
    
    for (Index i = 0; i < mGradientIn.rows(); i++)
    {
        for (Index j = 0; j < mGradientIn.cols(); j++)
        {
            float fStd = sqrtf(_batchVariance(j) + _fEpsilon);
            mGradientIn(i, j) = mGradNormalized(i, j) * (1.0f / fStd) +
                                mGradVariance(j) * (2.0f * mCentered(i, j) / iBatchSize) +
                                mGradMean(j) * (1.0f / iBatchSize);
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerBatchNormalization::set_momentum(float fMomentum)
{
    assert(fMomentum >= 0.0f && fMomentum <= 1.0f);
    _fMomentum = fMomentum;
}
///////////////////////////////////////////////////////////////////////////////
void LayerBatchNormalization::set_epsilon(float fEpsilon)
{
    assert(fEpsilon > 0.0f);
    _fEpsilon = fEpsilon;
}
///////////////////////////////////////////////////////////////////////////////
float LayerBatchNormalization::get_momentum() const
{
    return _fMomentum;
}
///////////////////////////////////////////////////////////////////////////////
float LayerBatchNormalization::get_epsilon() const
{
    return _fEpsilon;
}
///////////////////////////////////////////////////////////////////////////////
}