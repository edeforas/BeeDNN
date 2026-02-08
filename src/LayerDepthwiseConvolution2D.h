/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"

// Depthwise Convolution 2D Layer
// Reference: https://arxiv.org/abs/1704.04861 (MobileNets)
//
// In depthwise convolution, each input channel is convolved with a separate kernel
// This is much more efficient than standard convolution, reducing parameters by a factor of (kernel_height * kernel_width)
//
// Input shape: (batch_size, height, width, channels)
// Output shape: (batch_size, out_height, out_width, channels)
// Kernel shape per channel: (kernel_height, kernel_width, 1, 1)
// Total parameters: kernel_height * kernel_width * input_channels

namespace beednn {
class LayerDepthwiseConvolution2D : public Layer
{
public:
    explicit LayerDepthwiseConvolution2D(Index inRows, Index inCols, Index inChannels, 
                                        Index kernelRows, Index kernelCols, 
                                        Index rowStride = 1, Index colStride = 1);
    virtual ~LayerDepthwiseConvolution2D();
    virtual void init() override;

    virtual Layer* clone() const override;
    virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
    virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

    // Getter for layer parameters
    void get_params(Index& inRows, Index& inCols, Index& inChannels, 
                    Index& kernelRows, Index& kernelCols, Index& rowStride, Index& colStride) const;

private:
    // Input dimensions
    Index _inRows, _inCols, _inChannels;
    
    // Kernel dimensions
    Index _kernelRows, _kernelCols;
    
    // Stride
    Index _rowStride, _colStride;
    
    // Output dimensions (computed in init)
    Index _outRows, _outCols;
    
    // Kernel per channel: shape (kernel_rows * kernel_cols, in_channels)
    // Each column is the kernel for one input channel
    MatrixFloat _weight;
    
    // Bias per channel: shape (1, in_channels)
    MatrixFloat _bias;
    
    // Gradients
    MatrixFloat _gradientWeight;
    MatrixFloat _gradientBias;
    
    // Helper functions
    void computeOutputSize();
    void extractPatch(const MatrixFloat& mIn, Index iRow, Index iCol, Index iChannel, 
                     MatrixFloat& mPatch) const;
    void addGradientToPatch(MatrixFloat& mGradWeight, const MatrixFloat& mInputPatch, 
                           const float fGradient, Index iChannel) const;
};
}