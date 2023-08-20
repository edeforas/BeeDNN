/*
	Copyright (c) 2019, Etienne de Foras and the respective contributors
	All rights reserved.

	Use of this source code is governed by a MIT-style license that can be found
	in the LICENSE.txt file.
*/

#ifndef LayerGatedActivation_
#define LayerGatedActivation_

#include "Layer.h"
#include "Matrix.h"
namespace bee {
class Activation;

class LayerGatedActivation : public Layer
{
public:
	explicit LayerGatedActivation(const string& sActivation1, const string& sActivation2 = "Identity");
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
#endif
