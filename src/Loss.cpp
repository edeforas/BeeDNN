/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Loss.h"

////////////////////////////////////////////////////////////////
Loss::Loss()
{ }
////////////////////////////////////////////////////////////////
Loss::~Loss()
{ }
//////////////////////////////////////////////////////////////////////////////
class LossMeanSquareError : public Loss
{
public:
	string name() const override
	{
		return "MeanSquareError";
	}
	
	float compute(const MatrixFloat& mPredicted,const MatrixFloat& mTarget) const
	{
		if (mTarget.rows() == 0)
			return 0.f;

		return (mPredicted -mTarget ).cwiseAbs2().sum() / mTarget.rows();
	}
	
	float compute_derivate(const MatrixFloat& mPredicted,const MatrixFloat& mTarget) const
	{
		if (mTarget.rows() == 0)
			return 0.f;

		return (mPredicted - mTarget).sum() / mTarget.rows();
	}
};
//////////////////////////////////////////////////////////////////////////////
class LossCrossEntropy : public Loss   // as in: https://gombru.github.io/2018/05/23/cross_entropy_loss/
{
public:
	string name() const override
	{
		return "CrossEntropy";
	}
	
	float compute(const MatrixFloat& mPredicted,const MatrixFloat& mTarget) const
	{
		return 0.f; //todo

	}
	
	float compute_derivate(const MatrixFloat& mPredicted,const MatrixFloat& mTarget) const
	{
		return 0.f; //todo

	}
};
//////////////////////////////////////////////////////////////////////////////
Loss* get_loss(const string& sActivation)
{
    if(sActivation=="MeanSquareError")
        return new LossMeanSquareError;

    if(sActivation=="CrossEntropy")
        return new LossCrossEntropy;

    return nullptr;
}
//////////////////////////////////////////////////////////////////////////////
void list_loss_available(vector<string>& vsLoss)
{
    vsLoss.clear();

    vsLoss.push_back("MeanSquareError");
	vsLoss.push_back("CrossEntropy");
}
//////////////////////////////////////////////////////////////////////////////
