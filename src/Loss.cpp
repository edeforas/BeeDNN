/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Loss.h"

#include <cassert>

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
// as in :https://en.wikipedia.org/wiki/Hinge_loss
class LossHinge : public Loss
{
public:
	string name() const override
	{
		return "Hinge";
	}

	float compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		if (mTarget.size() == 0)
			return 0.f;

		//for now, element by element
		float fMean = 0.f;
		for (int i = 0; i < mTarget.size(); i++)
			fMean += std::max(0.f, 1.f - mTarget(i)*mPredicted(i));

		return fMean / mTarget.size();
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());
		
		mGradientLoss.resize(mTarget.rows(), mTarget.cols());

		for (int i = 0; i < mTarget.size(); i++)
		{
			float fProd = mTarget(i)*mPredicted(i);
			if (1. - fProd > 0.f)
			{
				mGradientLoss(i) = -mTarget(i);
			}
			else
				mGradientLoss(i) = 0.f;
		}
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
// from https://gombru.github.io/2018/05/23/cross_entropy_loss
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

        return -(mTarget.cwiseProduct(cwiseLog(mPredicted.cwiseMax(1.e-8f)))).mean(); //to avoid computing log(0)
	}
	
	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = -(mTarget.cwiseQuotient(mPredicted.cwiseMax(1.e-8f))); //to avoid computing 1/0
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

	if (sLoss == "Hinge")
		return new LossHinge;

	if (sLoss == "L2")
		return new LossL2;

	if (sLoss == "L1")
		return new LossL1;
	
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
//	vsLoss.push_back("Hinge");
	vsLoss.push_back("L2");
	vsLoss.push_back("L1");
	vsLoss.push_back("CategoricalCrossEntropy");
	vsLoss.push_back("BinaryCrossEntropy");
}
//////////////////////////////////////////////////////////////////////////////
