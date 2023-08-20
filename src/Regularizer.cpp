/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Regularizer.h"

#include <cassert>
#include <cmath>

using namespace std;
namespace bee {

//////////////////////////////////////////////////////////
Regularizer::Regularizer()
{
	_fParameter = -1.f; // by default
}
//////////////////////////////////////////////////////////
Regularizer::~Regularizer()
{}
//////////////////////////////////////////////////////////
void Regularizer::set_parameter(float fParameter)
{
	_fParameter = fParameter;
}
float Regularizer::get_parameter() const
{
	return _fParameter;
}
//////////////////////////////////////////////////////////
// this regularizer does nothing
class RegularizerNone : public Regularizer
{
public:
	RegularizerNone():Regularizer()
	{}

	~RegularizerNone() override
	{}

	string name() const override
	{
		return "None";
	}

	virtual void apply(MatrixFloat& w,MatrixFloat& dw) override
	{
		(void)w;
		(void)dw;
	}
};
//////////////////////////////////////////////////////////
class RegularizerL1 : public Regularizer
{
public:
	RegularizerL1() :Regularizer()
	{}

	~RegularizerL1() override
	{}

	string name() const override
	{
		return "L1";
	}

	virtual void set_parameter(float fParameter) override
	{
		if (fParameter != -1.f)
			_fParameter = fParameter;
		else
			_fParameter = 0.01f; // best value
	}

	virtual void apply(MatrixFloat& w, MatrixFloat& dw) override
	{
		dw = dw + w.cwiseSign()*_fParameter;
	}
};
//////////////////////////////////////////////////////////
class RegularizerL2 : public Regularizer
{
public:
	RegularizerL2() :Regularizer()
	{}

	~RegularizerL2() override
	{}

	string name() const override
	{
		return "L2";
	}

	virtual void set_parameter(float fParameter) override
	{
		if (fParameter != -1.f)
			_fParameter = fParameter;
		else
			_fParameter = 0.01f; // best value
	}

	virtual void apply(MatrixFloat& w, MatrixFloat& dw) override
	{
		dw = dw + w * _fParameter;
	}
};
//////////////////////////////////////////////////////////
// as in : https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L1L2

class RegularizerL1L2 : public Regularizer
{
public:
	RegularizerL1L2() :Regularizer()
	{}

	~RegularizerL1L2() override
	{}

	string name() const override
	{
		return "L1L2";
	}

	virtual void set_parameter(float fParameter) override
	{
		if (fParameter != -1.f)
			_fParameter = fParameter;
		else
			_fParameter = 0.01f; // best value
	}

	virtual void apply(MatrixFloat& w, MatrixFloat& dw) override
	{
		dw = dw + w * _fParameter + w.cwiseSign()*_fParameter;
	}
};
//////////////////////////////////////////////////////////
class RegularizerGradientClip : public Regularizer
{
public:	
    RegularizerGradientClip() :Regularizer()
    {}

    ~RegularizerGradientClip() override
    {}
	
	string name() const override
	{
		return "GradientClip";
	}

	virtual void set_parameter(float fParameter) override
	{
		if (fParameter != -1.f)
			_fParameter = fParameter;
		else
			_fParameter = 0.001f; // best value
	}

    virtual void apply(MatrixFloat& w,MatrixFloat& dw) override
    {
		(void)w;
		clamp(dw, -_fParameter, _fParameter);
    }
};
//////////////////////////////////////////////////////////
//  gradient norm clipping as in : https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48

class RegularizerGradientNormClip : public Regularizer
{
public:
	RegularizerGradientNormClip() :Regularizer()
	{}

	~RegularizerGradientNormClip() override
	{}

	string name() const override
	{
		return "GradientNormClip";
	}

	virtual void set_parameter(float fParameter) override
	{
		if (fParameter != -1.f)
			_fParameter = fParameter;
		else
			_fParameter = 0.01f; // best value
	}

	virtual void apply(MatrixFloat& w, MatrixFloat& dw) override
	{
		(void)w;

		float fNorm = dw.norm();
		if(fNorm> _fParameter)
			dw = (dw / fNorm)*_fParameter;
	}
};
//////////////////////////////////////////////////////////
class RegularizerGradientClipTanh : public Regularizer
{
public:
	RegularizerGradientClipTanh() :Regularizer()
	{}

	~RegularizerGradientClipTanh() override
	{}

	string name() const override
	{
		return "GradientClipTanh";
	}

	virtual void set_parameter(float fParameter) override
	{
		if (fParameter != -1.f)
			_fParameter = fParameter;
		else
			_fParameter = 0.001f; // best value
	}

	virtual void apply(MatrixFloat& w,MatrixFloat& dw) override
	{
		(void)w;
		dw =  ((dw* (1.f / _fParameter)).array().tanh())*_fParameter;
	}
};
//////////////////////////////////////////////////////////
Regularizer* create_regularizer(const string& sRegularizer)
{
	if (sRegularizer == "None")
		return new RegularizerNone;

	if (sRegularizer == "L1")
		return new RegularizerL1;

	if (sRegularizer == "L2")
		return new RegularizerL2;

	if (sRegularizer == "L1L2")
		return new RegularizerL1L2;

	if (sRegularizer == "GradientClip")
		return new RegularizerGradientClip;

	if (sRegularizer == "GradientNormClip")
		return new RegularizerGradientNormClip;

	if (sRegularizer == "GradientClipTanh")
		return new RegularizerGradientClipTanh;

	return nullptr;
}
//////////////////////////////////////////////////////////////////////////////
void list_regularizer_available(vector<string>& vsRegularizers)
{
    vsRegularizers.clear();

	vsRegularizers.push_back("None");
	vsRegularizers.push_back("L1");
	vsRegularizers.push_back("L2");
	vsRegularizers.push_back("L1L2");
	vsRegularizers.push_back("GradientClip");
	vsRegularizers.push_back("GradientNormClip");
	vsRegularizers.push_back("GradientClipTanh");
}
//////////////////////////////////////////////////////////////////////////////
}