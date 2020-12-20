/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerDropout_
#define LayerDropout_

#include "Layer.h"
#include "Matrix.h"

class LayerDropout : public Layer
{
public:
    explicit LayerDropout(float fRate);
    virtual ~LayerDropout() override;

    virtual Layer* clone() const override;

    virtual void predict(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

    float get_rate() const;

private:
	float _fRate;
	MatrixFloat _mask;
};

#endif
