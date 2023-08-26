/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Layer.h"

using namespace std;
namespace beednn {

////////////////////////////////////////////////////////////////
Layer::Layer(const string& sType):
_sType(sType)
{ 
	_bTrainMode = false;
	_bFirstLayer = false;

	_sWeightInitializer = "";
	_sBiasInitializer = "";
}
////////////////////////////////////////////////////////////////
Layer::~Layer()
{ }
////////////////////////////////////////////////////////////////
void Layer::init()
{ }
///////////////////////////////////////////////////////////////
string Layer::type() const
{
    return _sType;
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
bool Layer::has_weights() const
{
    return _weight.size()!=0.;
}
///////////////////////////////////////////////////////////////
vector<MatrixFloat*> Layer::weights()
{
	vector<MatrixFloat*> v;
	v.push_back(&_weight);
	return v;
}
///////////////////////////////////////////////////////////////
vector<MatrixFloat*> Layer::gradient_weights()
{
	vector<MatrixFloat*> v;
	v.push_back(&_gradientWeight);
	return v;
}
///////////////////////////////////////////////////////////////
bool Layer::has_biases() const
{
	return _bias.size() != 0.;
}
///////////////////////////////////////////////////////////////
vector<MatrixFloat*> Layer::biases()
{
	vector<MatrixFloat*> v;
	v.push_back(&_bias);
	return v;
}
///////////////////////////////////////////////////////////////
vector<MatrixFloat*> Layer::gradient_biases()
{
	vector<MatrixFloat*> v;
	v.push_back(&_gradientBias);
	return v;
}
///////////////////////////////////////////////////////////////
void Layer::set_weight_initializer(const string& sWeightInitializer)
{
	_sWeightInitializer = sWeightInitializer;
}
///////////////////////////////////////////////////////////////
void Layer::set_bias_initializer(const string& sBiasInitializer)
{
	_sBiasInitializer = sBiasInitializer;
}
///////////////////////////////////////////////////////////////
string Layer::weight_initializer() const
{
	return _sWeightInitializer;
}
///////////////////////////////////////////////////////////////
string Layer::bias_initializer() const
{
	return _sBiasInitializer;
}
///////////////////////////////////////////////////////////////
}