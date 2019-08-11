/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerDense.h"

#include <cmath> // for sqrt

///////////////////////////////////////////////////////////////////////////////
LayerDense::LayerDense(int iInSize, int iOutSize, bool bHasBias) :
    Layer(iInSize , iOutSize, "Dense"),
    _bHasBias(bHasBias)
{
    LayerDense::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDense::~LayerDense()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerDense::clone() const
{
    LayerDense* pLayer=new LayerDense(_iInSize,_iOutSize,_bHasBias);
    pLayer->_weight=_weight;
    return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::init()
{
	assert(_iInSize > 0);
	assert(_iOutSize > 0);
	
	_weight.resize(_iInSize+(_bHasBias?1:0),_iOutSize);

    //Xavier uniform initialization
    float a =sqrtf(6.f/(_iInSize+_iOutSize));
    _weight.setRandom();
    _weight*=a;

 //      if (_bHasBias)
 //          _weight.row(_iInSize).setZero(); //removed for now: accuracy is worse with bias initialized with zero

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::forward(const MatrixFloat& mMatIn,MatrixFloat& mMatOut) const
{
    if (_bHasBias)
        mMatOut = rowWiseAdd(mMatIn *_weight.topRows(_iInSize) , _weight.row(_iInSize)); //split _weight in [weightnobias, bias] in computation
    else
        mMatOut = mMatIn * _weight;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::backpropagation(const MatrixFloat &mInput,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    // optimisation: split _weight in [weightnobias, bias] in computation in cases of bias

    //backpropagation and computation of gradient
    if (_bHasBias)
    {
        _gradientWeight = ((addColumnOfOne(mInput)).transpose())*mGradientOut; //todo optimize

        mGradientIn = mGradientOut * _weight.topRows(_iInSize).transpose();
    }
    else
    {
        _gradientWeight = (mInput.transpose())*mGradientOut;

        mGradientIn = mGradientOut * (_weight.transpose());
    }
}
///////////////////////////////////////////////////////////////
bool LayerDense::has_bias() const
{
    return _bHasBias;
}
///////////////////////////////////////////////////////////////
