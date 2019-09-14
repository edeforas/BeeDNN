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
{
	_bClassBalancing = false;
}
////////////////////////////////////////////////////////////////
Loss::~Loss()
{ }
//////////////////////////////////////////////////////////////////////////////
void Loss::set_class_balancing(const MatrixFloat& mWeight)
{
	_mWeightBalancing = mWeight;
	_bClassBalancing = _mWeightBalancing.size() != 0;
}
//////////////////////////////////////////////////////////////////////////////
void Loss::balance_gradient_with_weight(const MatrixFloat& mTarget, MatrixFloat& mGradient) const
{
	if (!_bClassBalancing)
		return;

	for (int i = 0; i < mGradient.rows(); i++)
		mGradient.row(i) *= _mWeightBalancing((int)mTarget(i)); // todo check ?
}
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
		if(!_bClassBalancing)
		    return (mPredicted -mTarget).cwiseAbs2().mean();
		else
		{
			MatrixFloat mError = (mPredicted - mTarget).cwiseAbs2();
			balance_gradient_with_weight(mTarget, mError);
			return mError.mean();
		}
	}
	
	void compute_gradient(const MatrixFloat& mPredicted,const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = mPredicted - mTarget;
		balance_gradient_with_weight(mTarget, mGradientLoss);
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

        mGradientLoss=(mPredicted - mTarget).array().cwiseSign();
		balance_gradient_with_weight(mTarget, mGradientLoss);
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
		balance_gradient_with_weight(mTarget, mGradientLoss);
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
		balance_gradient_with_weight(mTarget, mGradientLoss);
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
		balance_gradient_with_weight(mTarget, mGradientLoss);
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
            float t = mTarget(i);
            fLoss += -(t*logf(max(p, 1.e-8f)));
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
            float t = mTarget(i);
            mGradientLoss(i) = -(t / max(p, 1.e-8f))+ (1.f - t)/(max(1.e-8f, 1.f - p));
		}
		//balance_gradient_with_weight(mTarget, mGradientLoss);
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
            float t = mTarget(i);
            fLoss += -(t*log(max(p, 1.e-8f)) + (1.f - t)*log(max(1.e-8f, 1.f - p)));
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
            float t = mTarget(i);
            mGradientLoss(i)= -(t / max(p,1.e-8f) - (1.f - t) / max((1.f - p),1.e-8f));
		}
		//balance_gradient_with_weight(mTarget, mGradientLoss);
	}
};
//////////////////////////////////////////////////////////////////////////////
Loss* create_loss(const string& sLoss)
{
    if(sLoss =="MeanSquaredError")
        return new LossMeanSquaredError;

    else if(sLoss =="MeanAbsoluteError")
        return new LossMeanAbsoluteError;

	else if(sLoss == "L2")
		return new LossL2;

	else if(sLoss == "L1")
		return new LossL1;

	else if(sLoss == "LogCosh")
		return new LossLogCosh;

	else if(sLoss =="CategoricalCrossEntropy")
        return new LossCategoricalCrossEntropy;

	else if(sLoss == "BinaryCrossEntropy")
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
