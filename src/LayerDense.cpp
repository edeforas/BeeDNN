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

    if (_bHasBias)
        _weight.row(_iInSize).setZero();

    Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
    if (_bHasBias)
        mOut = rowWiseAdd(mIn *_weight.topRows(_iInSize) , _weight.row(_iInSize)); //split _weight in [weightnobias, bias] in computation
    else
        mOut = mIn * _weight;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDense::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    if (_bHasBias)
    {
		// optimisation: split _weight in [weightnobias, bias]
		_gradientWeight = ((addColumnOfOne(mIn)).transpose())*mGradientOut*(1.f / mIn.rows()); //todo optimize
        
		if (_bFirstLayer)
			return;

		mGradientIn = mGradientOut * _weight.topRows(_iInSize).transpose();
    }
    else
    {
        _gradientWeight = (mIn.transpose())*mGradientOut*(1.f / mIn.rows());

		if (_bFirstLayer)
			return;

		mGradientIn = mGradientOut * (_weight.transpose());
    }
}
///////////////////////////////////////////////////////////////
bool LayerDense::has_bias() const
{
    return _bHasBias;
}
///////////////////////////////////////////////////////////////
