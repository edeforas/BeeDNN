/*
	Copyright (c) 2019, Etienne de Foras and the respective contributors
	All rights reserved.

	Use of this source code is governed by a MIT-style license that can be found
	in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"
namespace beednn {
class Activation;

class LayerGatedActivation : public Layer
{
public:
	explicit LayerGatedActivation(const std::string& sActivation1, const std::string& sActivation2 = "Identity");
	virtual ~LayerGatedActivation() override;

	virtual Layer* clone() const override;

	virtual void init() override;

	virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
	virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

private:
	Activation* _pActivation1;
	Activation* _pActivation2;
};
}
