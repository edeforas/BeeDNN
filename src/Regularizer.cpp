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
	_fVal= 1.f;
}
//////////////////////////////////////////////////////////
Regularizer::~Regularizer()
{}
//////////////////////////////////////////////////////////
void Regularizer::set_params(float fVal)
{
	_fVal = fVal;
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
		clamp(dw, -_fVal, _fVal);
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
		dw =  ((dw* (1.f / _fVal)).array().tanh())*_fVal;
	}
};
//////////////////////////////////////////////////////////
Regularizer* create_regularizer(const string& sRegularizer)
{
	if (sRegularizer == "Identiy")
		return new RegularizerIdentity;

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
	vsRegularizers.push_back("GradientClip");
	vsRegularizers.push_back("GradientClipTanh");
}
//////////////////////////////////////////////////////////////////////////////
