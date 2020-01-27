/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Net.h"
#include "Layer.h"

#include "Matrix.h"

#include "LayerActivation.h"
#include "LayerPRelu.h" 
#include "LayerSoftmax.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerUniformNoise.h"
#include "LayerGaussianNoise.h"
#include "LayerGaussianDropout.h"
#include "LayerGlobalGain.h"
#include "LayerGain.h"
#include "LayerGlobalBias.h"
#include "LayerBias.h"

#include "LayerPoolMax2D.h"
#include "LayerConvolution2D.h"
#include "LayerChannelBias.h"

#include <cmath>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
Net::Net()
{ 
    _bTrainMode = false;
    _iInputSize = 0;
	_bClassificationMode = true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Net::~Net()
{
    clear();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::clear()
{
    for(unsigned int i=0;i<_layers.size();i++)
        delete _layers[i];

    _layers.clear();
    _bTrainMode=false;
    _iInputSize=0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Net& Net::operator=(const Net& other)
{
    clear();

    for(unsigned int i=0;i<other._layers.size();i++)
        _layers.push_back(other._layers[i]->clone());

    _iInputSize= other._iInputSize;
	_bClassificationMode = other._bClassificationMode;

	return *this;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_dropout_layer(float fRatio)
{
    _layers.push_back(new LayerDropout(fRatio));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_gaussian_dropout_layer(float fProba)
{
    _layers.push_back(new LayerGaussianDropout(fProba));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_gaussian_noise_layer(float fStd)
{
	_layers.push_back(new LayerGaussianNoise(fStd));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_uniform_noise_layer(float fNoise)
{
	_layers.push_back(new LayerUniformNoise( fNoise));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_activation_layer(string sType)
{
    _layers.push_back(new LayerActivation(sType));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_prelu_layer()
{
	_layers.push_back(new LayerPRelu());
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_softmax_layer()
{
    _layers.push_back(new LayerSoftmax());
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_dense_layer(Index inSize,Index outSize,bool bHasBias)
{
    _layers.push_back(new LayerDense(inSize,outSize, bHasBias));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_globalgain_layer()
{
    _layers.push_back(new LayerGlobalGain());
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_gain_layer()
{
	_layers.push_back(new LayerGain());
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_globalbias_layer()
{
    _layers.push_back(new LayerGlobalBias());
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_bias_layer()
{
	_layers.push_back(new LayerBias());
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_poolmax2D_layer(Index iInRow, Index iInCols, Index iInChannels, Index iRowFactor, Index iColFactor)
{
	_layers.push_back(new LayerPoolMax2D(iInRow, iInCols, iInChannels, iRowFactor, iColFactor));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_convolution2D_layer(Index iInRows, Index iInCols, Index iInChannels, Index iKernelRows, Index iKernelCols, Index  iOutChannels, Index iRowStride, Index iColStride)
{
	_layers.push_back(new LayerConvolution2D(iInRows, iInCols, iInChannels, iKernelRows, iKernelCols, iOutChannels, iRowStride, iColStride));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::add_channel_bias_layer(Index iInRows, Index iInCols, Index iInChannels)
{
	_layers.push_back(new LayerChannelBias(iInRows, iInCols, iInChannels));
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
    MatrixFloat mTemp=mIn;
    for(unsigned int i=0;i<_layers.size();i++)
    {
        _layers[i]->forward(mTemp,mOut);
        mTemp=mOut; //todo avoid resize
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
void Net::set_classification_mode(bool bClassificationMode)
{
	_bClassificationMode = bClassificationMode;
}
/////////////////////////////////////////////////////////////////////////////////////////////
bool Net::is_classification_mode() const
{
	return _bClassificationMode;
}
/////////////////////////////////////////////////////////////////////////////////////////////
void Net::classify(const MatrixFloat& mIn, MatrixFloat& mClass) const
{
    MatrixFloat mOut;
	forward(mIn, mOut);
	
	if (mOut.cols() != 1)
		rowsArgmax(mOut, mClass); //one hot case
	else
	{
		mClass.resize(mOut.rows(), 1);
		for (int i = 0; i < mOut.rows(); i++)
			mClass(i, 0) = std::roundf(mOut(i, 0)); //categorical case
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::set_train_mode(bool bTrainMode)
{
    _bTrainMode = bTrainMode;

    for (unsigned int i = 0; i < _layers.size(); i++)
        _layers[i]->set_train_mode(bTrainMode);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
const vector<Layer*> Net::layers() const
{
    return _layers;
}
/////////////////////////////////////////////////////////////////////////////////////////////
Layer& Net::layer(size_t iLayer)
{
    return *(_layers[iLayer]);
}
/////////////////////////////////////////////////////////////////////////////////////////////
const Layer& Net::layer(size_t iLayer) const
{
	return *(_layers[iLayer]);
}
/////////////////////////////////////////////////////////////////////////////////////////////
size_t Net::size() const
{
    return _layers.size();
}
/////////////////////////////////////////////////////////////////////////////////////////////
void Net::init()
{
    if(_layers.empty())
        return;

    for(unsigned int i=0;i<_layers.size();i++)
        _layers[i]->init();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void Net::set_input_size(int iInputSize)
{
    _iInputSize=iInputSize;
	init();
}
/////////////////////////////////////////////////////////////////////////////////////////////
int Net::input_size() const
{
    return _iInputSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////