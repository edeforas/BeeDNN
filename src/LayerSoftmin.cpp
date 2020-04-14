/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSoftmin.h"

#include <cmath>

///////////////////////////////////////////////////////////////////////////////
LayerSoftmin::LayerSoftmin():
    Layer("Softmin")
{ }
///////////////////////////////////////////////////////////////////////////////
LayerSoftmin::~LayerSoftmin()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSoftmin::clone() const
{
    return new LayerSoftmin();
}
///////////////////////////////////////////////////////////////////////////////
void LayerSoftmin::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	MatrixFloat S;
	mOut=-mIn;

	for (Index r = 0; r < mOut.rows(); r++)// todo simplify and optimize
	{
		S = mOut.row(r); 
		S.array() -= S.maxCoeff(); //remove max
		S = S.array().exp();
		mOut.row(r) =S/ S.sum();
	}
}
///////////////////////////////////////////////////////////////////////////////
// from https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
void LayerSoftmin::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	if (_bFirstLayer)
		return;

    MatrixFloat S;
	mGradientIn.resizeLike(mGradientOut);

	for (Index r = 0; r < mIn.rows(); r++) // todo simplify and optimize
	{
		S = -mIn.row(r);
		S.array() -= S.maxCoeff(); //remove max
		S = S.array().exp();

		float s = S.sum();
		for (Index c = 0; c < S.cols(); c++)
		{
			float expx = S(c);
			mGradientIn(r, c) = -mGradientOut(r, c)*(expx*(s-expx)) / (s*s);
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
