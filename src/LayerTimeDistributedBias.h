/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerTimeDistributedBias_
#define LayerTimeDistributedBias_

#include "Layer.h"
#include "Matrix.h"
namespace bee {
class LayerTimeDistributedBias : public Layer
{
public:
    explicit LayerTimeDistributedBias(int iFrameSize, const std::string& sBiasInitializer = "Zeros");
    virtual ~LayerTimeDistributedBias();

    virtual Layer* clone() const override;

    int frame_size() const;
    virtual void init() override;
    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;
private:
	int _iFrameSize;
};
}
#endif
