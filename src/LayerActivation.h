/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerActivation_
#define LayerActivation_

#include "Layer.h"
#include "Matrix.h"

#include <string>

namespace beednn {
class Activation;

class LayerActivation : public Layer
{
public:
    explicit LayerActivation(const std::string& sActivation);
    virtual ~LayerActivation() override;

    virtual Layer* clone() const override;

    virtual void forward(const MatrixFloat& mIn, MatrixFloat &mOut) override;
	
    virtual void backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn) override;

private:
    Activation * _pActivation;
};
}
#endif
