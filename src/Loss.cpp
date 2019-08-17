/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Loss.h"

#include <cassert>
#include <cmath>


////////////////////////////////////////////////////////////////
Loss::Loss()
{ }
////////////////////////////////////////////////////////////////
Loss::~Loss()
{ }
//////////////////////////////////////////////////////////////////////////////
class LossMeanSquaredError : public Loss
{
public:
	string name() const override
	{
		return "MeanSquaredError";
	}
	
	float compute(const MatrixFloat& mPredicted,const MatrixFloat& mTarget) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		if (mTarget.size() == 0)
			return 0.f;

        return (mPredicted -mTarget ).cwiseAbs2().mean();
	}
	
	void compute_gradient(const MatrixFloat& mPredicted,const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = mPredicted - mTarget;
	}
};
//////////////////////////////////////////////////////////////////////////////
class LossMeanAbsoluteError : public Loss
{
public:
	string name() const override
	{
		return "MeanAbsoluteError";
	}
	
	float compute(const MatrixFloat& mPredicted,const MatrixFloat& mTarget) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		if (mTarget.size() == 0)
			return 0.f;

        return (mPredicted -mTarget ).cwiseAbs().mean();
	}
	
	void compute_gradient(const MatrixFloat& mPredicted,const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

        mGradientLoss=(mPredicted - mTarget).cwiseSign();
	}
};
//////////////////////////////////////////////////////////////////////////////
// same as MeanSquareError but do not divide by nbSamples
// see https://isaacchanghau.github.io/post/loss_functions/
class LossL2 : public Loss
{
public:
	string name() const override
	{
		return "L2";
	}

	float compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		if (mTarget.size() == 0)
			return 0.f;

		return (mPredicted - mTarget).cwiseAbs2().sum();
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = mPredicted - mTarget;
	}
};
//////////////////////////////////////////////////////////////////////////////
// same as MeanAbsoluteError but do not divide by nbSamples
// see https://isaacchanghau.github.io/post/loss_functions/
class LossL1 : public Loss
{
public:
	string name() const override
	{
		return "L1";
	}

	float compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		if (mTarget.size() == 0)
			return 0.f;

		return (mPredicted - mTarget).cwiseAbs().sum();
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = (mPredicted - mTarget).cwiseSign();
	}
};
//////////////////////////////////////////////////////////////////////////////

class LossLogCosh : public Loss
{
public:
	string name() const override
	{
		return "LogCosh";
	}

	float compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		if (mTarget.size() == 0)
			return 0.f;

		return (mPredicted - mTarget).array().cosh().log().sum();
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = (mPredicted - mTarget).array().tanh();
	}
};
//////////////////////////////////////////////////////////////////////////////
// from https://gombru.github.io/2018/05/23/cross_entropy_loss
// and : https://sefiks.com/2017/12/17/a-gentle-introduction-to-cross-entropy-loss-function/
class LossCategoricalCrossEntropy : public Loss
{
public:
	string name() const override
	{
		return "CategoricalCrossEntropy";
	}
	
	float compute(const MatrixFloat& mPredicted,const MatrixFloat& mTarget) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		float fLoss = 0.f;
		for (int i = 0; i < mTarget.size(); i++)
		{
			float p = mPredicted(i);
			float y = mTarget(i);
			fLoss += -(y*logf(max(p, 1.e-8f)));
		}
		return fLoss / mTarget.size();
	}
	
	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss.resize(mTarget.rows(), mTarget.cols());
		for (int i = 0; i < mTarget.size(); i++)
		{
			float p = mPredicted(i);
			float y = mTarget(i);
			mGradientLoss(i) = -(y / max(p, 1.e-8f))+ (1.f - y)/(max(1.e-8f, 1.f - p));
		}
	}
};
//////////////////////////////////////////////////////////////////////////////
// from https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
// and https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
class LossBinaryCrossEntropy : public Loss
{
public:
	string name() const override
	{
		return "BinaryCrossEntropy";
	}

	float compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget) const override
	{
		assert(mTarget.cols() == 1);
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());
		
		float fLoss = 0.f;
		for (int i = 0; i < mTarget.size(); i++)
		{
			float p = mPredicted(i);
			float y = mTarget(i);
			fLoss += -(y*log(max(p, 1.e-8f)) + (1.f - y)*log(max(1.e-8f, 1.f - p)));
		}
		return fLoss / mTarget.size();
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss.resize(mTarget.rows(), mTarget.cols());
		for (int i = 0; i < mTarget.size(); i++)
		{
			float p = mPredicted(i);
			float y = mTarget(i);
			mGradientLoss(i)= -(y / max(p,1.e-8f) - (1.f - y) / max((1.f - p),1.e-8f));
		}
	}
};
//////////////////////////////////////////////////////////////////////////////
Loss* create_loss(const string& sLoss)
{
    if(sLoss =="MeanSquaredError")
        return new LossMeanSquaredError;

    if(sLoss =="MeanAbsoluteError")
        return new LossMeanAbsoluteError;

	if (sLoss == "L2")
		return new LossL2;

	if (sLoss == "L1")
		return new LossL1;

	if (sLoss == "LogCosh")
		return new LossLogCosh;

	if(sLoss =="CategoricalCrossEntropy")
        return new LossCategoricalCrossEntropy;

	if (sLoss == "BinaryCrossEntropy")
		return new LossBinaryCrossEntropy;
	
	return nullptr;
}
//////////////////////////////////////////////////////////////////////////////
void list_loss_available(vector<string>& vsLoss)
{
    vsLoss.clear();

    vsLoss.push_back("MeanSquaredError");
	vsLoss.push_back("MeanAbsoluteError");
	vsLoss.push_back("L2");
	vsLoss.push_back("L1");
	vsLoss.push_back("LogCosh");
	vsLoss.push_back("CategoricalCrossEntropy");
	vsLoss.push_back("BinaryCrossEntropy");
}
//////////////////////////////////////////////////////////////////////////////
