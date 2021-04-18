/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerRNN.h"

///////////////////////////////////////////////////////////////////////////////
LayerRNN::LayerRNN(const string& sType) :
    Layer(sType)
{
    LayerRNN::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerRNN::~LayerRNN()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::init()
{
	_bias.resize(0,0);
	_weight.resize(0,0);

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if(_bias.size()==0)
	{
		_bias.setZero(1, mIn.cols());
		_weight.setOnes(1,mIn.cols());
	}

    mOut.resizeLike(mIn);

	for (int i = 0; i < mOut.rows(); i++)
		for (int j = 0; j < mOut.cols(); j++)
		{
			mOut(i,j) =mIn(i,j)* _weight(j) +_bias(j);
		}
}
///////////////////////////////////////////////////////////////////////////////
void LayerRNN::forward_one_sample(const MatrixFloat& mIn, MatrixFloat& mOut)
{

}
///////////////////////////////////////////////////////////////////////////////

void LayerRNN::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	_gradientBias = colWiseMean(mGradientOut);
	_gradientWeight = colWiseMean((mIn.transpose())*mGradientOut);

	if (_bFirstLayer)
		return;

	mGradientIn = mGradientOut;
	for (int i = 0; i < mGradientIn.rows(); i++)
		for (int j = 0; j < mGradientIn.cols(); j++)
		{
			mGradientIn(i, j) *= _weight(j);
		}
}
///////////////////////////////////////////////////////////////