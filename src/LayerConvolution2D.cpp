/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

//for now: no bias (add LayerBias just after this one gives same results), mode ='valid'

#include "LayerConvolution2D.h"

#include <cmath> // for sqrt
namespace bee{

///////////////////////////////////////////////////////////////////////////////
LayerConvolution2D::LayerConvolution2D(Index iInRows, Index iInCols, Index iInChannels, Index iKernelRows, Index iKernelCols, Index iOutChannels, Index iRowStride, Index iColStride) :
    Layer("Convolution2D")
{
	_iInRows = iInRows;
	_iInCols = iInCols;
	_iInChannels = iInChannels;
	_iSamples = 0;// set in predict()

	_iKernelRows = iKernelRows;
	_iKernelCols = iKernelCols;
	_iRowStride = iRowStride;
	_iColStride = iColStride; 

	_iOutChannels = iOutChannels;
	
	_iBorderRows=iKernelRows>>1;
	_iBorderCols=iKernelCols>>1;

	// out without strides
	_iOutRows=_iInRows-2* _iBorderRows;
	_iOutCols=_iInCols-2* _iBorderCols;

	//manage strides
	if(_iRowStride>1)
		_iOutRows = (_iOutRows + 1) / _iRowStride;
	if (_iColStride > 1)
		_iOutCols = (_iOutCols + 1) / _iColStride;

	create_im2col_LUT();
	LayerConvolution2D::init();

	fastLUT = true; 
}
///////////////////////////////////////////////////////////////////////////////
LayerConvolution2D::~LayerConvolution2D()
{ }
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::init()
{
	_weight.resize(_iOutChannels, _iKernelRows * _iKernelCols * _iInChannels);

	//Xavier uniform initialization
	float a = sqrtf(6.f / (_iInRows*_iInCols*_iInChannels + _iOutCols*_iOutRows*_iOutChannels));
	setRandomUniform(_weight);
	_weight *= a;

	_gradientWeight.resizeLike(_weight);
	_gradientWeight.setZero();
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::get_params(Index& iInRows, Index& iInCols, Index& iInChannels, Index& iKernelRows, Index& iKernelCols, Index& iOutChannels, Index& iRowStride, Index& iColStride) const
{
	iInRows = _iInRows;
	iInCols = _iInCols;
	iInChannels = _iInChannels;
	iKernelRows = _iKernelRows;
	iKernelCols = _iKernelCols;
	iRowStride = _iRowStride;
	iColStride = _iColStride;
	iOutChannels = _iOutChannels;
}
///////////////////////////////////////////////////////////////////////////////
Layer* LayerConvolution2D::clone() const
{
	LayerConvolution2D* pLayer = new LayerConvolution2D(_iInRows, _iInCols, _iInChannels,_iKernelRows,_iKernelCols,_iOutChannels,_iRowStride,_iColStride);
	pLayer->_weight = _weight;
	pLayer->_gradientWeight = _gradientWeight;
	return pLayer;
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::forward(const MatrixFloat& mIn,MatrixFloat& mOut)
{
	if(fastLUT)
		im2col_LUT(mIn, _im2colT); //optimized
	else
		im2col(mIn, _im2colT); //slow

	mOut = _weight * (_im2colT.transpose());// optimized GEMM product with transposed
	reshape_to_out(mOut);
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::backpropagation(const MatrixFloat &mIn,const MatrixFloat &mGradientOut, MatrixFloat &mGradientIn)
{
	(void)mIn;
	assert(mGradientOut.rows() == _iSamples);
	assert(mGradientOut.cols() == _iOutRows * _iOutCols*_iOutChannels);
	
	MatrixFloat mGradientUnflat = mGradientOut;
	reshape_from_out(mGradientUnflat);

	_gradientWeight = mGradientUnflat *_im2colT;

	assert(_gradientWeight.rows() == _weight.rows());
	assert(_gradientWeight.cols() == _weight.cols());

	if (_bFirstLayer)
		return;

	MatrixFloat mGradientCol= _weight.transpose()*mGradientUnflat;

	if(fastLUT)
		col2im_LUT(mGradientCol, mGradientIn); //faster
	else
		col2im(mGradientCol, mGradientIn); //slow

	assert(mGradientIn.rows() == mIn.rows());
	assert(mGradientIn.cols() == mIn.cols());
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::im2col(const MatrixFloat & mIn, MatrixFloat & mCol)
{
	//slow reference version	
	assert(mIn.cols() == _iInRows * _iInCols*_iInChannels);
	_iSamples = (int)mIn.rows();

	mCol.resize(_iOutRows * _iOutCols* _iSamples,_iKernelRows * _iKernelCols*_iInChannels );
	
	for (Index iSample = 0; iSample < _iSamples; iSample++)
	{
		for (Index iInChannel = 0; iInChannel < _iInChannels; iInChannel++)
		{
			for (Index iKRow = 0; iKRow < _iKernelRows; iKRow++)
			{
				for (Index iKCol = 0; iKCol < _iKernelCols; iKCol++)
				{
					for (Index iOutRow = 0; iOutRow < _iOutRows; iOutRow++)
					{
						for (Index iOutCol = 0; iOutCol < _iOutCols; iOutCol++)
						{
							Index iRowInPlane = iOutRow*_iRowStride+iKRow;
							Index iColInPlane = iOutCol*_iColStride+iKCol;
							
							assert(iRowInPlane >=0);
							assert(iColInPlane >=0);
							assert(iRowInPlane < _iInRows);
							assert(iColInPlane < _iInCols);
							
							float f = mIn(iSample, iInChannel*_iInRows*_iInCols + iRowInPlane*_iInCols+ iColInPlane);
							mCol(
								iSample*_iOutCols*_iOutRows + iOutRow * _iOutCols + iOutCol
								,
								iInChannel*_iKernelRows * _iKernelCols + iKRow * _iKernelCols + iKCol
							) = f;
						}
					}
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::col2im(const MatrixFloat & mCol, MatrixFloat & mIm)
{
	assert(mCol.rows() == _iKernelRows * _iKernelCols*_iInChannels);
	assert(mCol.cols() == _iOutRows * _iOutCols* _iSamples);

	//slow reference version
	mIm.setZero(_iSamples, _iInChannels* _iInRows * _iInCols);
	for (Index iSample = 0; iSample < _iSamples; iSample++)
	{
		for (Index iOutRow = 0; iOutRow < _iOutRows; iOutRow++)
		{
			for (Index iOutCol = 0; iOutCol < _iOutCols; iOutCol++)
			{
				for (Index iInChannel = 0; iInChannel < _iInChannels; iInChannel++)
				{
					for (Index iKRow = 0; iKRow < _iKernelRows; iKRow++)
					{
						for (Index iKCol = 0; iKCol < _iKernelCols; iKCol++)
						{
							Index iRowInPlane = iOutRow * _iRowStride + iKRow;
							Index iColInPlane = iOutCol * _iColStride + iKCol;

							assert(iRowInPlane >= 0);
							assert(iColInPlane >= 0);
							assert(iRowInPlane < _iInRows);
							assert(iColInPlane < _iInCols);

							float f = mCol(
								iInChannel*_iKernelRows * _iKernelCols + iKRow * _iKernelCols + iKCol,
								iSample*_iOutCols*_iOutRows + iOutRow * _iOutCols + iOutCol
								);
							mIm(
								iSample,
								iInChannel*_iInRows*_iInCols + iRowInPlane * _iInCols + iColInPlane)
								+= f;
						}
					}
				}
			}
		}
	}

	//rescale data to compute mean instead of sum
	mIm *= (1.f / (_iKernelRows* _iKernelCols* _iOutChannels));
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::col2im_LUT(const MatrixFloat & mCol, MatrixFloat & mIm)
{
	assert(mCol.rows() == _iKernelRows * _iKernelCols*_iInChannels);
	assert(mCol.cols() == _iOutRows * _iOutCols* _iSamples);

	//slow reference version
	mIm.setZero(_iSamples, _iInChannels* _iInRows * _iInCols);
	for (Index iSample = 0; iSample < _iSamples; iSample++)
	{
		for (Index iOutRow = 0; iOutRow < _iOutRows; iOutRow++)
		{


				/*
				Index iDecal = iSample * _iInRows*_iInCols*_iInChannels + iOutRow * _iInCols *_iRowStride + iOutCol * _iColStride;
				const float *pIn = mIn.data() + iDecal;
				Index iRowDest = iOutCol + iOutRow * _iOutCols;
				float * pOut = mCol.data() + iRowDest * mCol.cols() + iSample * _iOutCols*_iOutRows*iLUTRows;
				for (Index iLUT = 0; iLUT < iLUTRows; iLUT++)
					*(pOut + iLUT) = *(pIn + _im2ColLUT[iLUT]);
				*/

				Index iFromCol=iSample*_iOutCols*_iOutRows + iOutRow * _iOutCols ;
				Index iDecal = iSample * _iInRows*_iInCols*_iInChannels+ + iOutRow * _iRowStride*_iInCols;
				float *pfIm = mIm.data() + iDecal;

				for (Index iInChannel = 0; iInChannel < _iInChannels; iInChannel++)
				{
					for (Index iKRow = 0; iKRow < _iKernelRows; iKRow++)
					{
						Index iRowInPlane = iKRow*_iInCols+ iInChannel * _iInRows*_iInCols;

						for (Index iKCol = 0; iKCol < _iKernelCols; iKCol++)
						{

							for (Index iOutCol = 0; iOutCol < _iOutCols; iOutCol++)
							{


							float f = mCol(
								iInChannel*_iKernelRows * _iKernelCols + iKRow * _iKernelCols + iKCol,
								iFromCol + iOutCol
							);

							*(pfIm + iRowInPlane + iKCol+ iOutCol * _iColStride) += f;
						}
					}
				}

			}
		}
	}

	//rescale data to compute mean instead of sum
	mIm *= (1.f / (_iKernelRows* _iKernelCols* _iOutChannels));
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::reshape_to_out(MatrixFloat & mOut)
{
	assert(mOut.cols() == _iSamples * _iOutRows * _iOutCols);
	assert(mOut.rows() ==  _iOutChannels);

	mOut.resize(_iSamples * _iOutChannels, _iOutRows * _iOutCols );
	_tempImg = mOut; //for now, use a copy

	for (Index iOutChannel = 0; iOutChannel < _iOutChannels; iOutChannel++)
	{
		for (Index iSample = 0; iSample < _iSamples; iSample++)
		{
			Index iOrigRow = iSample+ iOutChannel* _iSamples;
			Index iDestRow = iOutChannel+ iSample* _iOutChannels;

			mOut.row(iDestRow) = _tempImg.row(iOrigRow);
		}
	}
	mOut.resize(_iSamples, _iOutRows * _iOutCols * _iOutChannels);
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::reshape_from_out(MatrixFloat & mOut)
{
	assert(mOut.cols() == _iOutChannels*_iOutRows * _iOutCols);
	assert(mOut.rows() == _iSamples);

	mOut.resize(_iSamples * _iOutChannels, _iOutRows * _iOutCols);

	_tempImg = mOut; //for now, use a copy

	for (Index iOutChannel = 0; iOutChannel < _iOutChannels; iOutChannel++)
	{
		for (Index iSample = 0; iSample < _iSamples; iSample++)
		{
			Index iOrigRow = iSample + iOutChannel * _iSamples;
			Index iDestRow = iOutChannel + iSample * _iOutChannels;

			mOut.row(iOrigRow) = _tempImg.row(iDestRow);
		}
	}
	mOut.resize(_iOutChannels, _iOutRows*_iOutCols*_iSamples);
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::im2col_LUT(const MatrixFloat & mIn, MatrixFloat & mCol)
{
	assert(mIn.cols() == _iInRows * _iInCols*_iInChannels);
	_iSamples = (int)mIn.rows();
	mCol.resize(_iOutRows * _iOutCols* _iSamples, _iKernelRows * _iKernelCols*_iInChannels);

	Index iLUTRows = _im2ColLUT.size();

//#pragma omp parallel for
	for (Index iSample = 0; iSample < _iSamples; iSample++)
	{
		for (Index iOutRow = 0; iOutRow < _iOutRows; iOutRow++)
		{
			for (Index iOutCol = 0; iOutCol < _iOutCols; iOutCol++)
			{
				Index iDecal = iSample * _iInRows*_iInCols*_iInChannels + iOutRow * _iInCols *_iRowStride + iOutCol* _iColStride;

				const float *pfIn = mIn.data() + iDecal;
				Index iRowDest = iOutCol + iOutRow * _iOutCols;
				float * pfCol = mCol.data() + iRowDest * mCol.cols()+iSample*_iOutCols*_iOutRows*iLUTRows;
				for (Index iLUT = 0; iLUT < iLUTRows; iLUT++)
				{
					*(pfCol +iLUT)= *(pfIn+_im2ColLUT[iLUT]);
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
void LayerConvolution2D::create_im2col_LUT()
{
	_im2ColLUT.resize(_iKernelRows * _iKernelCols*_iInChannels);
	for (Index iInChannel = 0; iInChannel < _iInChannels; iInChannel++)
	{
		for (Index iKRow = 0; iKRow < _iKernelRows; iKRow++)
		{
			for (Index iKCol = 0; iKCol < _iKernelCols; iKCol++)
			{
				_im2ColLUT[iKCol + iKRow * _iKernelCols+iInChannel* _iKernelRows*_iKernelCols] = 
					(iInChannel*_iInRows*_iInCols+iKRow*_iInCols+iKCol);
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////
}