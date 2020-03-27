/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Regularizer.h"

#include <cassert>
#include <cmath>

//////////////////////////////////////////////////////////
Regularizer::Regularizer()
{
	_fParameter = 0.f;
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
class RegularizerIdentity : public Regularizer
{
public:
	RegularizerIdentity():Regularizer()
	{}

	~RegularizerIdentity() override
	{}

	string name() const override
	{
		return "Identity";
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

	virtual void apply(MatrixFloat& w, MatrixFloat& dw) override
	{
		dw = dw + w * _fParameter;
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

    virtual void apply(MatrixFloat& w,MatrixFloat& dw) override
    {
		(void)w;
		clamp(dw, -_fParameter, _fParameter);
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

	virtual void apply(MatrixFloat& w,MatrixFloat& dw) override
	{
		(void)w;
		dw =  ((dw* (1.f / _fParameter)).array().tanh())*_fParameter;
	}
};
//////////////////////////////////////////////////////////
Regularizer* create_regularizer(const string& sRegularizer)
{
	if (sRegularizer == "Identiy")
		return new RegularizerIdentity;

	if (sRegularizer == "L1")
		return new RegularizerL1;

	if (sRegularizer == "L2")
		return new RegularizerL2;

	if (sRegularizer == "GradientClip")
		return new RegularizerGradientClip;

	if (sRegularizer == "GradientClipTanh")
		return new RegularizerGradientClipTanh;

	return nullptr;
}
//////////////////////////////////////////////////////////////////////////////
void list_regularizer_available(vector<string>& vsRegularizers)
{
    vsRegularizers.clear();

	vsRegularizers.push_back("Identity");
	vsRegularizers.push_back("L1");
	vsRegularizers.push_back("L2");
	vsRegularizers.push_back("GradientClip");
	vsRegularizers.push_back("GradientClipTanh");
}
//////////////////////////////////////////////////////////////////////////////
