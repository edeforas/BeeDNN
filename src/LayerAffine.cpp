/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerAffine.h"

///////////////////////////////////////////////////////////////////////////////
LayerAffine::LayerAffine() :
    Layer("Affine")
{
    LayerAffine::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerAffine::~LayerAffine()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerAffine::clone() const
{
    LayerAffine* pLayer=new LayerAffine();
	pLayer->_weight = _weight;
	pLayer->_bias = _bias;

    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerAffine::init()
{
	_bias.resize(0,0);
	_weight.resize(0,0);

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerAffine::predict(const MatrixFloat& mIn,MatrixFloat& mOut)
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
void LayerAffine::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
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