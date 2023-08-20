/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/
#pragma once

#include "Layer.h"
#include "Matrix.h"

class LayerTimeDistributedDot : public Layer
{
public:
    explicit LayerTimeDistributedDot(int iInFrameSize,int iOutFrameSize, const string& sWeightInitializer = "GlorotUniform");
    virtual ~LayerTimeDistributedDot();

    virtual Layer* clone() const override;
    
    int in_frame_size() const;
    int out_frame_size() const;
    virtual void init() override;
    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;
	
private:
	int _iInFrameSize, _iOutFrameSize;
};

