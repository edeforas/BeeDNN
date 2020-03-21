/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Net.h"
#include "Layer.h"

#include "Matrix.h"

#include <cmath>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////
Net::Net()
{ 
    _bTrainMode = false;
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
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Net& Net::operator=(const Net& other)
{
    clear();

    for(size_t i=0;i<other._layers.size();i++)
        _layers.push_back(other._layers[i]->clone());

    _bClassificationMode = other._bClassificationMode;

    return *this;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
// add the layer, take the ownership of the layer 
void Net::add(Layer* l)
{
	_layers.push_back(l);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
// replace a layer, take the ownership of the layer
void Net::replace(size_t iLayer, Layer* l)
{
	assert(iLayer < _layers.size());

	delete _layers[iLayer];
	_layers[iLayer] = l;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void Net::forward(const MatrixFloat& mIn,MatrixFloat& mOut) const
{
	//todo cut in mini batches so save memory
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
    for(unsigned int i=0;i<_layers.size();i++)
        _layers[i]->init();
}
/////////////////////////////////////////////////////////////////////////////////////////////