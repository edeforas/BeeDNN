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
    mOut=cwiseExp(mIn);

    for(int i=0;i<mOut.rows();i++)
        mOut.row(i)/=mOut.row(i).sum();

 //   arraySub(mOut,mIn.maxCoeff()); // todo simplify and optimize
}
///////////////////////////////////////////////////////////////////////////////
void LayerSoftmax::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mDelta, MatrixFloat &mNewDelta)
{
    MatrixFloat S;
    forward(mInput,S);
    mNewDelta.resize(mDelta.rows(),mDelta.cols());

    for(int i=0;i<S.rows();i++)
    {
        MatrixFloat sR=S.row(i);
		MatrixFloat mDiag(sR.cols(), sR.cols());
		for (int iC = 0; iC < sR.cols(); iC++)
			mDiag(iC, iC) = sR(0, iC);

		MatrixFloat m3 = (mDiag - (sR.transpose()*sR));
        MatrixFloat m4= mDelta.row(i) * m3;
		mNewDelta.row(i) = m4;
    }
}
///////////////////////////////////////////////////////////////////////////////
