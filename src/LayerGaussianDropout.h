/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerGaussianDropout_
#define LayerGaussianDropout_

#include "Layer.h"
#include "Matrix.h"

#include <string>
using namespace std;

class Activation;

class LayerGaussianDropout : public Layer
{
public:
    LayerGaussianDropout(int iSize,float fProba);
    virtual ~LayerGaussianDropout() override;

    virtual Layer* clone() const override;

    virtual void init() override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;

    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

    float get_proba() const;

private:
    void create_mask();

    float _fProba;
    float _fStdev;
	MatrixFloat _mask;
};

#endif
