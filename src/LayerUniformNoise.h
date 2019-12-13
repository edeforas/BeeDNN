/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerUniformNoise_
#define LayerUniformNoise_

#include "Layer.h"
#include "Matrix.h"

#include <random>
using namespace std;

class Activation;

class LayerUniformNoise : public Layer
{
public:
    LayerUniformNoise(int iSize,float fNoise);
    virtual ~LayerUniformNoise() override;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

    float get_noise() const;

private:
    float _fNoise;
	uniform_real_distribution<float> _distUniform;

};

#endif
