/*
	Copyright (c) 2019, Etienne de Foras and the respective contributors
	All rights reserved.

	Use of this source code is governed by a MIT-style license that can be found
	in the LICENSE.txt file.
*/

//example of Gated Activation:
// GLU as in :https://arxiv.org/abs/1612.08083

#include "LayerGatedActivation.h"
#include "Activations.h"

///////////////////////////////////////////////////////////////////////////////
LayerGatedActivation::LayerGatedActivation(const string& sActivation1, const string& sActivation2) :
	Layer("GatedActivation")
{
	_pActivation1 = get_activation(sActivation1);
	_pActivation2 = get_activation(sActivation2);
	LayerGatedActivation::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGatedActivation::~LayerGatedActivation()
{
	delete _pActivation1;
	delete _pActivation2;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGatedActivation::clone() const
{
	return new LayerGatedActivation(_pActivation1->name(), _pActivation2->name());
}
///////////////////////////////////////////////////////////////////////////////
void LayerGatedActivation::init()
{
	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGatedActivation::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	assert(((mIn.cols() & 1) == 0) && "mIn must have an even size");

	Index iNbCols = mIn.cols();
	Index iNbColsHalf = iNbCols / 2;

	mOut.resize(mIn.rows(), iNbColsHalf);
	for (int r = 0; r < mIn.rows(); r++)
		for (int c = 0; c < iNbColsHalf; c++)
			mOut(r, c) = _pActivation1->apply(mIn(r, c)) * _pActivation2->apply(mIn(r, c + iNbColsHalf));
}
///////////////////////////////////////////////////////////////////////////////
void LayerGatedActivation::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
	if (_bFirstLayer)
		return;

	mGradientIn.resizeLike(mIn);
	Index iNbCols = mIn.cols();
	Index iNbColsHalf = iNbCols / 2;

	// 1st part with activation and 2nd part without activation
	for (int r = 0; r < mIn.rows(); r++)
		for (int c = 0; c < iNbColsHalf; c++)
		{
			float g = mGradientOut(r, c);
			mGradientIn(r, c) = g * _pActivation2->apply(mIn(r, c + iNbColsHalf)) * _pActivation1->derivation(mIn(r, c)); // (dL/dt)*g(y)*f'(x1)*g(x2)
			mGradientIn(r, c + iNbColsHalf) = g * _pActivation1->apply(mIn(r, c)) * _pActivation2->derivation(mIn(r, c + iNbColsHalf)); // (dL/dt)*f(x1)*g'(x2)
		}
}
///////////////////////////////////////////////////////////////