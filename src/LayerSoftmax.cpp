/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerSoftmax.h"

#include "Activation.h"

///////////////////////////////////////////////////////////////////////////////
LayerSoftmax::LayerSoftmax():
    Layer(0,0,"Softmax")
{ }
///////////////////////////////////////////////////////////////////////////////
LayerSoftmax::~LayerSoftmax()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerSoftmax::clone() const
{
    return new LayerSoftmax();
}
///////////////////////////////////////////////////////////////////////////////
void LayerSoftmax::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
    mOut=mIn;
    arraySub(mOut,mIn.maxCoeff()); // todo simplify and optimize
    mOut=cwiseExp(mOut);
    mOut/=mOut.sum();
}
///////////////////////////////////////////////////////////////////////////////
void LayerSoftmax::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta)
{
    //WIP WIP
    MatrixFloat S;
    forward(mInput,S); //WIP

    (void)(mDelta); // todo

	MatrixFloat sd(S.cols(), S.cols());
	sd.setZero();

	//sd.diagonal() = S;
	for (int i = 0; i < S.cols(); i++)
		sd(i, i) = S(0, i);

	
//	= S.asDiagonal();



    mNewDelta=sd-(S.transpose()*S);


//todo
}
///////////////////////////////////////////////////////////////////////////////
