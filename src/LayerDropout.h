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

#include <string>
#include <random>
using namespace std;

class Activation;

class LayerDropout : public Layer
{
public:
    LayerDropout(int iSize,float fRate);
    virtual ~LayerDropout() override;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;

    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

    float get_rate() const;

private:
	float _fRate;
	MatrixFloat _mask;

	default_random_engine _RNGgenerator;
	bernoulli_distribution _distBernoulli;
};

#endif
