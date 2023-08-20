/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// SwiGLU as in : https://kikaben.com/swiglu-2020/

#include "LayerSwiGLU.h"
#include "Activations.h"
namespace bee{

///////////////////////////////////////////////////////////////////////////////
LayerSwiGLU::LayerSwiGLU() :
    Layer("SwiGLU")
{
	_pActivation = get_activation("Swish");
    LayerSwiGLU::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerSwiGLU::~LayerSwiGLU()
{
	delete _pActivation;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSwiGLU::clone() const
{
    return new LayerSwiGLU();
}
///////////////////////////////////////////////////////////////////////////////
void LayerSwiGLU::init()
{
    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerSwiGLU::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	assert( (mIn.cols() % 1) == 0); // mIn must have an even size

	Index iNbCols=mIn.cols();
	Index iNbColsHalf= iNbCols/2;

	mOut.resize(mIn.rows(), iNbColsHalf);
	for (int r = 0; r < mIn.rows(); r++)
		for (int c = 0; c < iNbColsHalf; c++)
			mOut(r, c) = _pActivation->apply(mIn(r, c))*mIn(r, c + iNbColsHalf);
}
///////////////////////////////////////////////////////////////////////////////
void LayerSwiGLU::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
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
}