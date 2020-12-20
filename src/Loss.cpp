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
void Loss::balance_with_weight(const MatrixFloat& mTarget, MatrixFloat& mGradient) const
{
	assert(_bClassBalancing);
		
	for (int i = 0; i < mGradient.rows(); i++)
	{
		assert(mTarget(i)>=0);
		assert(mTarget(i)<_mWeightBalancing.size());
		
		mGradient.row(i) *= _mWeightBalancing((int)mTarget(i));
	}
}
//////////////////////////////////////////////////////////////////////////////
class LossMeanSquaredError : public Loss
{
public:
	string name() const override
	{
		return "MeanSquaredError";
	}
	
	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());
		/*
		if (mTarget.size() == 0)
		{
			mLoss.resize(0, 0);
			return ;
		}
		*/
		mLoss= colWiseMean( (mPredicted - mTarget).cwiseAbs2() );
		/*
		if (_bClassBalancing)
		{
			MatrixFloat mError = (mPredicted - mTarget).cwiseAbs2();
			balance_with_weight(mTarget, mError);
			return mError.mean();
		}
		*/
	}
	
	void compute_gradient(const MatrixFloat& mPredicted,const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = (mPredicted - mTarget)/ (float)mPredicted.cols();
/*        if(_bClassBalancing)
			balance_with_weight(mTarget, mGradientLoss);
*/	}
};

//////////////////////////////////////////////////////////////////////////////
//Huber Loss from https://en.wikipedia.org/wiki/Huber_loss
#define HUBER_SIGMA (1.f)
class LossHuber : public Loss
{
public:
	string name() const override
	{
		return "Huber";
	}

	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		MatrixFloat m = (mPredicted - mTarget).cwiseAbs();

		for (Index i = 0; i < m.size(); i++)
		{
			if (m(i) < HUBER_SIGMA)
				m(i) = 0.5f*(m(i) * m(i));
			else
				m(i) = HUBER_SIGMA * m(i) - 0.5f*HUBER_SIGMA * HUBER_SIGMA;
		}

		mLoss= colWiseMean(m);
		/*
		if (!_bClassBalancing)
			return m.mean();
		else
		{
			MatrixFloat mError = m;
			balance_with_weight(m, mError);
			return mError.mean();
		}
		*/
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = mPredicted- mTarget;
		for (Index i = 0; i < mGradientLoss.size(); i++)
		{
			if (mGradientLoss(i) > HUBER_SIGMA)
				mGradientLoss(i) = HUBER_SIGMA;
			else if (mGradientLoss(i) < -HUBER_SIGMA)
				mGradientLoss(i) = -HUBER_SIGMA;
		}
		/*
		if (_bClassBalancing)
			balance_with_weight(mTarget, mGradientLoss);
	*/
	}
};

//////////////////////////////////////////////////////////////////////////////
//PseudoHuber Loss from https://en.wikipedia.org/wiki/Huber_loss
#define PSEUDOHUBER_SIGMA (1.f)
#define PSEUDOHUBER_SIGMA_2 (PSEUDOHUBER_SIGMA*PSEUDOHUBER_SIGMA)
#define PSEUDOHUBER_INV_SIGMA_2 (1.f/PSEUDOHUBER_SIGMA_2)
class LossPseudoHuber : public Loss
{
public:
	string name() const override
	{
		return "PseudoHuber";
	}

	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		MatrixFloat m = (mPredicted - mTarget).cwiseAbs2();

		for (Index i = 0; i < m.size(); i++)
			m(i) = PSEUDOHUBER_SIGMA_2 * (sqrtf(m(i)*PSEUDOHUBER_INV_SIGMA_2 + 1.f) - 1.f);

		mLoss = colWiseMean(m);
		/*
		if (!_bClassBalancing)
			return m.mean();
		else
		{
			MatrixFloat mError = m;
			balance_with_weight(m, mError);
			return mError.mean();
		}
		*/
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = mPredicted - mTarget;

		for (Index i = 0; i < mGradientLoss.size(); i++)
		{
			float x = mGradientLoss(i);
			mGradientLoss(i) = x / sqrtf(x*x*PSEUDOHUBER_INV_SIGMA_2 + 1.f);
		}
		/*
		if (_bClassBalancing)
			balance_with_weight(mTarget, mGradientLoss);
	*/
	}
};
//////////////////////////////////////////////////////////////////////////////
class LossMeanCubicError : public Loss
{
public:
	string name() const override
	{
		return "MeanCubicError";
	}

	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mLoss = colWiseMean((mPredicted - mTarget).array().cube().cwiseAbs());
			/*
		if (!_bClassBalancing)
			return (mPredicted - mTarget).array().cube().cwiseAbs().mean();
		else
		{
			MatrixFloat mError = (mPredicted - mTarget).array().cube().cwiseAbs();
			balance_with_weight(mTarget, mError);
			return mError.mean();
		}
		*/
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = mPredicted - mTarget;

		//todo vectorize
		for (Index i = 0; i < mGradientLoss.size(); i++)
		{
			float x = mGradientLoss(i);
			mGradientLoss(i) = 3.f*x*x*(x > 0.f ? 1.f : -1.f);
		}
		/*
		if (_bClassBalancing)
			balance_with_weight(mTarget, mGradientLoss);
	*/
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
	
	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mLoss = colWiseMean((mPredicted - mTarget).cwiseAbs());
		/*
		if(!_bClassBalancing)
		    return (mPredicted -mTarget ).cwiseAbs().mean();
		else
		{
			MatrixFloat mError = (mPredicted - mTarget).cwiseAbs();
			balance_with_weight(mTarget, mError);
			return mError.mean();
		}
		*/
	}
	
	void compute_gradient(const MatrixFloat& mPredicted,const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

        mGradientLoss=(mPredicted - mTarget).array().cwiseSign();
    /*    if(_bClassBalancing)
			balance_with_weight(mTarget, mGradientLoss);
	*/
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

	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());
		/*
		if (mTarget.size() == 0)
		{
			mLoss.resize(0, 0);
			return;
		}
		*/
		mLoss = colWiseSum((mPredicted - mTarget).cwiseAbs2());
		/*
		if (_bClassBalancing)
		{
			MatrixFloat mError = (mPredicted - mTarget).cwiseAbs2();
			balance_with_weight(mTarget, mError);
			return mError.mean();
		}
		*/
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = (mPredicted - mTarget);
		/*        if(_bClassBalancing)
					balance_with_weight(mTarget, mGradientLoss);
		*/
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

	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mLoss = colWiseSum((mPredicted - mTarget).cwiseAbs());
		/*
		if(!_bClassBalancing)
		    return (mPredicted -mTarget ).cwiseAbs().sum();
		else
		{
			MatrixFloat mError = (mPredicted - mTarget).cwiseAbs();
			balance_with_weight(mTarget, mError);
			return mError.sum();
		}
		*/
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = (mPredicted - mTarget).cwiseSign() / (float)mPredicted.cols();

		/*
        if(_bClassBalancing)
			balance_with_weight(mTarget, mGradientLoss);
	*/
	}
};
//////////////////////////////////////////////////////////////////////////////
// same as MeanCubicError but do not divide by nbSamples
class LossL3 : public Loss
{
public:
	string name() const override
	{
		return "L3";
	}

	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mLoss = colWiseSum((mPredicted - mTarget).array().cube().cwiseAbs());
		/*
		if (!_bClassBalancing)
			return (mPredicted - mTarget).array().cube().cwiseAbs().sum();
		else
		{
			MatrixFloat mError = (mPredicted - mTarget).array().cube().cwiseAbs();
			balance_with_weight(mTarget, mError);
			return mError.sum();
		}
		*/
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = mPredicted - mTarget;
		
		//todo vectorize
		for (Index i = 0; i < mGradientLoss.size(); i++)
		{
			float x = mGradientLoss(i);
			mGradientLoss(i) = 3.f*x*x*(x > 0.f ? 1.f:-1.f);
		}
		/*
		if (_bClassBalancing)
			balance_with_weight(mTarget, mGradientLoss);
	*/
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

	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mLoss = colWiseMean((mPredicted - mTarget).array().cosh().log());
		/*
		if(!_bClassBalancing)
		    return (mPredicted - mTarget).array().cosh().log().sum();
		else
		{
			MatrixFloat mError = (mPredicted - mTarget).array().cosh().log();
			balance_with_weight(mTarget, mError);
			return mError.sum();
		}*/
	}

	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss = (mPredicted - mTarget).array().tanh();
		/*
        if(_bClassBalancing)
			balance_with_weight(mTarget, mGradientLoss);
	*/
	}
};
//////////////////////////////////////////////////////////////////////////////
// from https://gombru.github.io/2018/05/23/cross_entropy_loss
// and https://sefiks.com/2017/12/17/a-gentle-introduction-to-cross-entropy-loss-function/
// and https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
class LossCategoricalCrossEntropy : public Loss
{
public:
	string name() const override
	{
		return "CategoricalCrossEntropy"; //truth is one hot encoded
	}
	
	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());

		Index r = mPredicted.rows();
		Index c = mPredicted.cols();
		mLoss.resize(r, 1);

		for (Index j = 0; j < r; j++)
		{
			float fLoss = 0.f;
			for (int i = 0; i < c; i++)
			{
				float t = mTarget(j, i);
				float p = mPredicted(j, i);
				float lossTmp = -(t*logf(max(p, 1.e-8f)));
				fLoss += lossTmp;
			}
			mLoss(j, 0) = fLoss;
		}
	}

			/*
            if(_bClassBalancing)
                lossTmp*=_mWeightBalancing((int)mTarget(i));

		}



		}
	
		return fLoss / mTarget.size();
	}
	*/
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
		/*
        if(_bClassBalancing)
            balance_with_weight(mTarget, mGradientLoss);
    */
	}
};
//////////////////////////////////////////////////////////////////////////////
// and https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
class LossSparseCategoricalCrossEntropy : public Loss
{
public:
	string name() const override
	{
		return "SparseCategoricalCrossEntropy"; //truth is index encoded
	}
	
	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == 1);
		assert(mTarget.rows() == mPredicted.rows());

		Index r = mPredicted.rows();
		Index c = mPredicted.cols();
		mLoss.resize(r, 1);
		/*
		for (Index j = 0; j < r; j++)
		{
			float fLoss = 0.f;
			for (int i = 0; i < c; i++)
			{

				float t = mTarget(j, i);
				float p = mPredicted(j, i);
				float lossTmp = -(t*logf(max(p, 1.e-8f)));
				fLoss += lossTmp;
			}
			mLoss(j, 0) = fLoss;
		}
		*/

		

		float fLoss = 0.f;
		for (int r = 0; r< mPredicted.rows(); r++)
		{	
			float lossTmp=-logf(max(mPredicted(r,(int)mTarget(r)), 1.e-8f)); //computing only when truth=1.
			/*
			if(_bClassBalancing)
				lossTmp*=_mWeightBalancing((int)mTarget(r));
				*/

			mLoss(r, 1) = lossTmp;

		//	fLoss+=lossTmp;
		}

	//	return fLoss / mTarget.size();
	}
		
	void compute_gradient(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mGradientLoss) const override
	{ //todo optimize all
		assert(mTarget.cols() == 1);
		assert(mTarget.rows() == mPredicted.rows());

		mGradientLoss=mPredicted;
		
		for (int r = 0; r < mGradientLoss.rows(); r++)
		{
            int t = (int)mTarget(r);
			for (int c = 0; c < mGradientLoss.cols(); c++)
			{
				float p = mGradientLoss(r, c);
				if(t==c)
					mGradientLoss(r,c) = -(1.f / max(p, 1.e-8f));
				else
					mGradientLoss(r,c) = 1.f/(max(1.e-8f, 1.f - p)); 
			}
		}
		/*
		if(_bClassBalancing)
            balance_with_weight(mTarget, mGradientLoss);
    */
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

	void compute(const MatrixFloat& mPredicted, const MatrixFloat& mTarget, MatrixFloat& mLoss) const override
	{
		assert(mTarget.cols() == 1);
		assert(mTarget.cols() == mPredicted.cols());
		assert(mTarget.rows() == mPredicted.rows());
		
		Index  r= mPredicted.rows();
		Index  c = mPredicted.cols();
		mLoss.resize(r, 1);

      //  float fLoss = 0.f;
		for (int i = 0; i < r; i++)
		{
			float p = mPredicted(i);
            float t = mTarget(i);
            float lossTmp=-(t*log(max(p, 1.e-8f)) + (1.f - t)*log(max(1.e-8f, 1.f - p)));

        //    if(_bClassBalancing)
         //       lossTmp*=_mWeightBalancing((int)mTarget(i));
			mLoss(i, 0) = lossTmp;

            //fLoss += lossTmp;
		}
		//return fLoss / mTarget.size();
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
		/*
        if(_bClassBalancing)
            balance_with_weight(mTarget, mGradientLoss);
	*/
	}
};

//////////////////////////////////////////////////////////////////////////////
Loss* create_loss(const string& sLoss)
{
    if(sLoss =="MeanSquaredError")
        return new LossMeanSquaredError;
	
    else if(sLoss =="MeanAbsoluteError")
        return new LossMeanAbsoluteError;
		
	if (sLoss == "MeanCubicError")
		return new LossMeanCubicError;
		
	else if(sLoss == "L2")
		return new LossL2;
	
	else if(sLoss == "L1")
		return new LossL1;

	else if (sLoss == "L3")
		return new LossL3;

	else if(sLoss == "LogCosh")
		return new LossLogCosh;
		
	if (sLoss == "Huber")
		return new LossHuber;
	
	if (sLoss == "PseudoHuber")
		return new LossPseudoHuber;
		
	else if(sLoss =="CategoricalCrossEntropy")
        return new LossCategoricalCrossEntropy;

	else if(sLoss =="SparseCategoricalCrossEntropy")
        return new LossSparseCategoricalCrossEntropy;

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
	vsLoss.push_back("MeanCubicError");
	vsLoss.push_back("L2");
	vsLoss.push_back("L1");
	vsLoss.push_back("L3");
	vsLoss.push_back("Huber");
	vsLoss.push_back("PseudoHuber");
	vsLoss.push_back("LogCosh");
	vsLoss.push_back("CategoricalCrossEntropy");
	vsLoss.push_back("SparseCategoricalCrossEntropy");
	vsLoss.push_back("BinaryCrossEntropy");
}
//////////////////////////////////////////////////////////////////////////////
