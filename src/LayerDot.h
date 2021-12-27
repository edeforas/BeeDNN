/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerDot_
#define LayerDot_

#include "Layer.h"
#include "Matrix.h"

class LayerDot : public Layer
{
public:
    LayerDot(Index iInputSize,Index iOutputSize, string sWeightInitializer = "GlorotUniform");
    virtual ~LayerDot() override;

	Index input_size() const;
	Index output_size() const;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;

    virtual void init() override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

private:
	Index _iInputSize, _iOutputSize;
};

#endif
