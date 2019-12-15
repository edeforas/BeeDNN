/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Layer.h"

////////////////////////////////////////////////////////////////
Layer::Layer(const string& sType):
_iInputSize(0),
_iOutputSize(0),
_sType(sType)
{ 
	_bTrainMode = false;
	_bFirstLayer = false;

	_iInputRows = 0;
	_iInputCols = 0;
	_iInputPlanes = 0;

	_iOutputRows = 0;
	_iOutputCols = 0;
	_iOutputPlanes = 0;
}
////////////////////////////////////////////////////////////////
Layer::~Layer()
{ }
////////////////////////////////////////////////////////////////
void Layer::init()
{ }
////////////////////////////////////////////////////////////////
void Layer::set_shape(int iInputRows, int iInputCols, int iInputPlanes, int& iOutputRows, int& iOutputCols, int & iOutputPlanes)
{ 
	_iInputRows = iInputRows;
	_iInputCols = iInputCols;
	_iInputPlanes = iInputPlanes;

	iOutputRows = iInputRows;
	iOutputCols = iInputCols;
	iOutputPlanes = iInputPlanes;

	_iOutputRows = iOutputRows;
	_iOutputCols = iOutputCols;
	_iOutputPlanes = iOutputPlanes;

	_iInputSize = iInputRows * iInputCols*iInputPlanes;
	_iOutputSize = iOutputRows * iOutputCols*iOutputPlanes;
}
///////////////////////////////////////////////////////////////
string Layer::type() const
{
    return _sType;
}
///////////////////////////////////////////////////////////////
int Layer::in_size() const
{
    return _iInputSize;
}
///////////////////////////////////////////////////////////////
int Layer::out_size() const
{
    return _iOutputSize;
}
///////////////////////////////////////////////////////////////
void Layer::set_first_layer(bool bFirstLayer)
{
	_bFirstLayer = bFirstLayer;
}
///////////////////////////////////////////////////////////////
void Layer::set_train_mode(bool bTrainMode)
{
	_bTrainMode = bTrainMode;
}
///////////////////////////////////////////////////////////////
bool Layer::has_weight() const
{
    return false; //not learnable by default
}
///////////////////////////////////////////////////////////////
MatrixFloat& Layer::weights()
{
    return _weight;
}
///////////////////////////////////////////////////////////////
MatrixFloat& Layer::gradient_weights()
{
    return _gradientWeight;
}
///////////////////////////////////////////////////////////////
