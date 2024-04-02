/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerDepthwiseConvolution2D.h"
#include <cassert>
#include <cmath>

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerDepthwiseConvolution2D::LayerDepthwiseConvolution2D(Index inRows, Index inCols, Index inChannels, 
                                                         Index kernelRows, Index kernelCols, 
                                                         Index rowStride, Index colStride) :
    Layer("DepthwiseConvolution2D"),
    _inRows(inRows),
    _inCols(inCols),
    _inChannels(inChannels),
    _kernelRows(kernelRows),
    _kernelCols(kernelCols),
    _rowStride(rowStride),
    _colStride(colStride)
{
    assert(inRows > 0 && inCols > 0 && inChannels > 0);
    assert(kernelRows > 0 && kernelCols > 0);
    assert(rowStride > 0 && colStride > 0);
    assert(kernelRows <= inRows && kernelCols <= inCols);
    
    LayerDepthwiseConvolution2D::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDepthwiseConvolution2D::~LayerDepthwiseConvolution2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerDepthwiseConvolution2D::computeOutputSize()
{
    // Standard convolution output size formula
    // out_size = floor((in_size - kernel_size) / stride) + 1
    _outRows = (_inRows - _kernelRows) / _rowStride + 1;
    _outCols = (_inCols - _kernelCols) / _colStride + 1;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDepthwiseConvolution2D::init()
{
    computeOutputSize();
    
    // Initialize weights: shape (kernel_rows * kernel_cols, in_channels)
    // Each column is the depthwise kernel for one input channel
    Index iKernelSize = _kernelRows * _kernelCols;
    _weight.setRandom(iKernelSize, _inChannels);
    
    // Scale weights (Xavier initialization)
    float fScale = sqrtf(6.0f / (iKernelSize + _inChannels));
    for (Index i = 0; i < _weight.size(); i++)
    {
        _weight(i) = _weight(i) * fScale;
    }
    
    // Initialize bias: shape (1, in_channels)
    _bias.setZero(1, _inChannels);
    
    // Initialize gradients
    _gradientWeight.setZero(iKernelSize, _inChannels);
    _gradientBias.setZero(1, _inChannels);

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerDepthwiseConvolution2D::clone() const
{
    LayerDepthwiseConvolution2D* pLayer = new LayerDepthwiseConvolution2D(_inRows, _inCols, _inChannels,
                                                                           _kernelRows, _kernelCols,
                                                                           _rowStride, _colStride);
    pLayer->_weight = _weight;
    pLayer->_bias = _bias;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDepthwiseConvolution2D::extractPatch(const MatrixFloat& mIn, Index iRow, Index iCol, Index iChannel, 
                                               MatrixFloat& mPatch) const
{
    // Extract a kernel_rows x kernel_cols patch from input at position (iRow, iCol) for channel iChannel
    // Input is stored as: [sample, height * width * channels]
    // Channel layout: row-major order
    
    mPatch.setZero(_kernelRows * _kernelCols, 1);
    
    for (Index kr = 0; kr < _kernelRows; kr++)
    {
        for (Index kc = 0; kc < _kernelCols; kc++)
        {
            Index iInRow = iRow + kr;
            Index iInCol = iCol + kc;
            
            // Compute linear index for this position
            Index iInIdx = (iInRow * _inCols + iInCol) * _inChannels + iChannel;
            Index iPatchIdx = kr * _kernelCols + kc;
            
            mPatch(iPatchIdx) = mIn(iInIdx);
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerDepthwiseConvolution2D::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
    assert(mIn.cols() == _inRows * _inCols * _inChannels);
    
    Index iBatchSize = mIn.rows();
    Index iOutSize = _outRows * _outCols * _inChannels;
    
    mOut.setZero(iBatchSize, iOutSize);
    
    Index iKernelSize = _kernelRows * _kernelCols;
    
    // For each sample in the batch
    for (Index iBatch = 0; iBatch < iBatchSize; iBatch++)
    {
        // For each output position
        for (Index iOutRow = 0; iOutRow < _outRows; iOutRow++)
        {
            for (Index iOutCol = 0; iOutCol < _outCols; iOutCol++)
            {
                // Compute input position
                Index iInRow = iOutRow * _rowStride;
                Index iInCol = iOutCol * _colStride;
                
                // For each channel (depthwise)
                for (Index iChan = 0; iChan < _inChannels; iChan++)
                {
                    // Extract patch for this channel
                    MatrixFloat mPatch(_kernelRows * _kernelCols, 1);
                    extractPatch(mIn, iInRow, iInCol, iChan, mPatch);
                    
                    // Compute convolution: dot product with kernel
                    float fConv = 0.0f;
                    for (Index iKIdx = 0; iKIdx < iKernelSize; iKIdx++)
                    {
                        fConv += mPatch(iKIdx) * _weight(iKIdx, iChan);
                    }
                    
                    // Add bias
                    fConv += _bias(iChan);
                    
                    // Store output
                    Index iOutIdx = (iOutRow * _outCols + iOutCol) * _inChannels + iChan;
                    mOut(iBatch, iOutIdx) = fConv;
                }
            }
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
void LayerDepthwiseConvolution2D::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
    assert(mIn.cols() == _inRows * _inCols * _inChannels);
    assert(mGradientOut.cols() == _outRows * _outCols * _inChannels);
    
    Index iBatchSize = mIn.rows();
    Index iKernelSize = _kernelRows * _kernelCols;
    
    // Initialize gradients
    _gradientWeight.setZero(iKernelSize, _inChannels);
    _gradientBias.setZero(1, _inChannels);
    mGradientIn.setZero(iBatchSize, mIn.cols());
    
    // For each sample in the batch
    for (Index iBatch = 0; iBatch < iBatchSize; iBatch++)
    {
        // For each output position
        for (Index iOutRow = 0; iOutRow < _outRows; iOutRow++)
        {
            for (Index iOutCol = 0; iOutCol < _outCols; iOutCol++)
            {
                // Compute input position
                Index iInRow = iOutRow * _rowStride;
                Index iInCol = iOutCol * _colStride;
                
                // For each channel (depthwise)
                for (Index iChan = 0; iChan < _inChannels; iChan++)
                {
                    Index iOutIdx = (iOutRow * _outCols + iOutCol) * _inChannels + iChan;
                    float fGradOut = mGradientOut(iBatch, iOutIdx);
                    
                    // Bias gradient
                    _gradientBias(iChan) += fGradOut;
                    
                    // Extract patch for this channel
                    MatrixFloat mPatch(_kernelRows * _kernelCols, 1);
                    extractPatch(mIn, iInRow, iInCol, iChan, mPatch);
                    
                    // Weight gradient: dL/dW = dL/dOut * dOut/dW = fGradOut * mPatch
                    for (Index iKIdx = 0; iKIdx < iKernelSize; iKIdx++)
                    {
                        _gradientWeight(iKIdx, iChan) += fGradOut * mPatch(iKIdx);
                    }
                    
                    // Input gradient: dL/dIn = dL/dOut * dOut/dIn = fGradOut * W
                    for (Index kr = 0; kr < _kernelRows; kr++)
                    {
                        for (Index kc = 0; kc < _kernelCols; kc++)
                        {
                            Index iInIdx = ((iInRow + kr) * _inCols + (iInCol + kc)) * _inChannels + iChan;
                            Index iKIdx = kr * _kernelCols + kc;
                            
                            mGradientIn(iBatch, iInIdx) += fGradOut * _weight(iKIdx, iChan);
                        }
                    }
                }
            }
        }
    }
    
    // Average gradients over batch
    float fBatchScale = 1.0f / iBatchSize;
    _gradientWeight = _gradientWeight * fBatchScale;
    _gradientBias = _gradientBias * fBatchScale;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDepthwiseConvolution2D::get_params(Index& inRows, Index& inCols, Index& inChannels, 
                                              Index& kernelRows, Index& kernelCols, Index& rowStride, Index& colStride) const
{
    inRows = _inRows;
    inCols = _inCols;
    inChannels = _inChannels;
    kernelRows = _kernelRows;
    kernelCols = _kernelCols;
    rowStride = _rowStride;
    colStride = _colStride;
}
///////////////////////////////////////////////////////////////////////////////
}