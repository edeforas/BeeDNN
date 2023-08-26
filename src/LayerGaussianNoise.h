/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"

#include <random>

namespace beednn {
class LayerGaussianNoise : public Layer
{
public:
    explicit LayerGaussianNoise(float fNoise);
    virtual ~LayerGaussianNoise() override;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

    float get_noise() const;

private:
    float _fNoise;
	std::normal_distribution<float> _distNormal;
};
}
