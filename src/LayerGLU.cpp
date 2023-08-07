/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// GLU as in :https://arxiv.org/abs/1612.08083

#include "LayerGLU.h"
#include "Activations.h"

///////////////////////////////////////////////////////////////////////////////
LayerGLU::LayerGLU() :
    Layer("GLU")
{
	_pActivation = get_activation("Sigmoid");
    LayerGLU::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerGLU::~LayerGLU()
{
	delete _pActivation;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerGLU::clone() const
{
    return new LayerGLU();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGLU::init()
{
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerGLU::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	assert(((mIn.cols() & 1) == 0) && "mIn must have an even size");

	Index iNbCols=mIn.cols();
	Index iNbColsHalf= iNbCols/2;

	mOut.resize(mIn.rows(), iNbColsHalf);
	for (int r = 0; r < mIn.rows(); r++)
		for (int c = 0; c < iNbColsHalf; c++)
			mOut(r, c) = _pActivation->apply(mIn(r, c))*mIn(r, c + iNbColsHalf);
}
///////////////////////////////////////////////////////////////////////////////
void LayerGLU::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
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
			mGradientIn(r, c) = mGradientOut(r, c) * mIn(r, c + iNbColsHalf) * _pActivation->derivation(mIn(r, c)); // (dL/dt)*g(y)*f'(x)
			mGradientIn(r, c + iNbColsHalf) = mGradientOut(r, c) * _pActivation->apply(mIn(r, c)); // (dL/dt)*f(x)*g'(y)
		}
}
///////////////////////////////////////////////////////////////