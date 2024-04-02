/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerDotMAM.h"

#include "Initializers.h"

using namespace std;
namespace beednn {

///////////////////////////////////////////////////////////////////////////////
LayerDotMAM::LayerDotMAM(Index iInputSize, Index iOutputSize,const string& sWeightInitializer) :
    Layer( "DotMAM"),
	_iInputSize(iInputSize),
	_iOutputSize(iOutputSize)
{
	set_weight_initializer(sWeightInitializer);
	LayerDotMAM::init();
}
///////////////////////////////////////////////////////////////////////////////
LayerDotMAM::~LayerDotMAM()
{ }
///////////////////////////////////////////////////////////////////////////////
Layer* LayerDotMAM::clone() const
{
    LayerDotMAM* pLayer=new LayerDotMAM(_iInputSize, _iOutputSize);
    pLayer->_weight=_weight;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerDotMAM::init()
{
	if (_iInputSize == 0)
		return;

	if (_iOutputSize == 0)
		return;

	assert(_iInputSize > 0);
	assert(_iOutputSize > 0);
	
	Initializers::compute(weight_initializer(), _weight, _iInputSize, _iOutputSize);
	
	Layer::init();
}
///////////////////////////////////////////////////////////////////////////////
void LayerDotMAM::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	//not optimized yet
	mOut.resize( mIn.rows(), _iOutputSize);
	if(_bTrainMode)
	{
		_mMinIndex.resizeLike(mOut); //index to selected input min data
		_mMaxIndex.resizeLike(mOut); //index to selected input max data
	}
	for(Index sample =0; sample < mIn.rows(); sample++)
	{
		for (Index colOut = 0; colOut < mOut.cols(); colOut++)
		{
			for (Index weightRow = 0; weightRow < _weight.rows(); weightRow++)
			{
				float fMax = -1.e38f;
				float fMin = +1.e38f;
				Index iPosMinIn = -1;
				Index iPosMaxIn = -1;

				for (Index c = 0; c < mIn.cols(); c++)
				{
					float fSample = mIn(sample, c)*_weight(c, colOut);
					if (fSample > fMax)
					{
						fMax = fSample;
						iPosMaxIn = c;
					}

					if (fSample < fMin)
					{
						fMin = fSample;
						iPosMinIn = c;
					}
				}

				mOut(sample, colOut) = fMax + fMin;
				if (_bTrainMode)
				{
					_mMinIndex(sample, colOut) = (float)iPosMinIn;
					_mMaxIndex(sample, colOut) = (float)iPosMaxIn;
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerDotMAM::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
    (void)mIn;

	_gradientWeight.setZero(_weight.rows(), _weight.cols());
	for (Index sample = 0; sample < mGradientOut.rows(); sample++)
	{
		for (Index c = 0; c < mGradientOut.cols(); c++)
		{
			Index minIndex = (int)_mMinIndex(sample, c);
			Index maxIndex = (int)_mMaxIndex(sample, c);
			float fGradout = mGradientOut(sample, c);

			_gradientWeight(minIndex, c) += _weight(minIndex, c)*fGradout;
			_gradientWeight(maxIndex, c) += _weight(maxIndex, c)*fGradout;
		}
	}

	if (_bFirstLayer)
		return;

	// compute mGradientIn
	mGradientIn = mGradientOut * (_weight.transpose()); // todo check
}
///////////////////////////////////////////////////////////////
Index LayerDotMAM::input_size() const
{
	return _iInputSize;
}
///////////////////////////////////////////////////////////////
Index LayerDotMAM::output_size() const
{
	return _iOutputSize;
}
///////////////////////////////////////////////////////////////
}