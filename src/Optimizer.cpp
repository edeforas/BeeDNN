/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Optimizer.h"

#include <cassert>
#include <cmath>

// Network optimizer as in:
// http://cs231n.github.io/neural-networks-3/#update

//////////////////////////////////////////////////////////
Optimizer::Optimizer()
{
	_fLearningRate= -1.f;
	_fDecay= -1.f;
	_fMomentum = -1.f;
}
//////////////////////////////////////////////////////////
Optimizer::~Optimizer()
{}
//////////////////////////////////////////////////////////
void Optimizer::set_params(float fLearningRate, float fDecay, float fMomentum)  //-1.f is for default params
{
	_fLearningRate = fLearningRate;
	_fDecay = fDecay;
	_fMomentum = fMomentum;

	init();
}
//////////////////////////////////////////////////////////
class OptimizerSGD : public Optimizer
{
public:	
    OptimizerSGD()
    {}

    ~OptimizerSGD() override
    {}
	
	string name() const override
	{
		return "SGD";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.01f;
	}

    virtual void optimize(MatrixFloat& w,const MatrixFloat& dw) override
    {
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

        // Vanilla update
        //	x += -learning_rate * dx
        w -=  dw * _fLearningRate ;
    }
};
//////////////////////////////////////////////////////////
class OptimizerMomentum : public Optimizer
{
public:
    OptimizerMomentum()
    {}

    ~OptimizerMomentum() override
    {}

	string name() const override
	{
		return "Momentum";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.01f;
		if (_fMomentum == -1.f) _fMomentum = 0.9f;

        _v.resize(0,0);
    }

    virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
    {
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

        // init _V if needed
        if (_v.size() == 0)
            _v.setZero(w.rows(), w.cols());

        // v = mu * v - learning_rate * dx // integrate velocity
        // x += v // integrate position
        _v = _v*_fMomentum - dw*_fLearningRate;

        w += _v;
    }
private:
    MatrixFloat _v;
};
//////////////////////////////////////////////////////////
class OptimizerNesterov : public Optimizer
{
public:
    OptimizerNesterov()
    {}

    ~OptimizerNesterov() override
    {}

	string name() const override
	{
		return "Nesterov";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.01f;
		if (_fMomentum == -1.f) _fMomentum = 0.9f;

        _v.resize(0, 0);
    }

    virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
    {
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

        // init _V if needed
        if (_v.size() == 0)
            _v.setZero(w.rows(), w.cols());

        // v_prev = v # back this up
        // v = mu * v - learning_rate * dx # velocity update stays the same
        // x += -mu * v_prev + (1 + mu) * v # position update changes form
        _v_prev = _v;
        _v = _v*_fMomentum - dw*_fLearningRate ;
        w += _v_prev*(_fMomentum) + _v*(1.f + _fMomentum) ;
    }
private:
    MatrixFloat _v, _v_prev;
};
//////////////////////////////////////////////////////////////////////////////
class OptimizerAdagrad : public Optimizer
{
public:
    OptimizerAdagrad()
    {}

    ~OptimizerAdagrad() override
    {}

	string name() const override
	{
		return "Adagrad";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.01f;

        _cache.resize(0,0);
    }

    virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
    {
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

        // init _cache if needed
        if (_cache.size() == 0)
            _cache.setZero(dw.rows(), dw.cols());

        // cache += dx**2
        // x += - learning_rate * dx / (np.sqrt(cache) + eps)

        _cache +=dw.cwiseAbs2();
        w += dw.cwiseQuotient(_cache.cwiseSqrt().cwiseMax(1.e-8f))*(-_fLearningRate);
    }
private:
    MatrixFloat _cache;
};
//////////////////////////////////////////////////////////
class OptimizerRMSProp : public Optimizer
{
public:
    OptimizerRMSProp()
    {}

    ~OptimizerRMSProp() override
    {}

	string name() const override
	{
		return "RMSProp";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.01f;
		if (_fDecay == -1.f) _fDecay = 0.99f;

		_cache.resize(0,0);
    }

    virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
    {
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

        // init _cache if needed
        if (_cache.size() == 0)
            _cache.setZero(dw.rows(), dw.cols());

        // cache = decay_rate * cache + (1 - decay_rate) * dx**2
        // x += - learning_rate * dx / (np.sqrt(cache) + eps)
        _cache =_cache*_fDecay+dw.cwiseAbs2()*(1.f-_fDecay);
        w += dw.cwiseQuotient(_cache.cwiseSqrt().cwiseMax(1.e-8f))*(-_fLearningRate);
    }
private:
    MatrixFloat _cache;
};
//////////////////////////////////////////////////////////
class OptimizerAdam : public Optimizer
{
public:
    OptimizerAdam()
    {
        beta1=0.9f;
        beta2=0.999f;
		beta1_prod = 0.f;
		beta2_prod = 0.f;
    }

    ~OptimizerAdam() override
    {}

	string name() const override
	{
		return "Adam";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.001f;

        _m.resize(0,0);
        _v.resize(0,0);

        beta1_prod=beta1;
        beta2_prod=beta2;
    }

    virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
    {
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

        // init _m and _v if needed
        if (_v.size() == 0)
        {
            _m.setZero(dw.rows(), dw.cols());
            _v.setZero(dw.rows(), dw.cols());
        }

        // Adam, with first step bias correction
        //m = beta1*m + (1-beta1)*dx
        //v = beta2*v + (1-beta2)*(dx**2)
        //x += - learning_rate/(1-beta1_prod) * m / (np.sqrt(v/(1-beta2_prod)) + eps)
        //beta1_prod*=beta1;
        //beta2_prod*=beta2;

        _m=_m*beta1+dw*(1.f-beta1);
        _v=_v*beta2+dw.cwiseAbs2()*(1.f-beta2);
        w += _m.cwiseQuotient((_v/(1.f-beta2_prod)).cwiseSqrt().cwiseMax(1.e-8f))*(-_fLearningRate/(1.f-beta1_prod));
        beta1_prod*=beta1;
        beta2_prod*=beta2;
    }
private:
    MatrixFloat _m, _v;
    float beta1, beta2, beta1_prod, beta2_prod;
};
//////////////////////////////////////////////////////////
class OptimizerNadam : public Optimizer
{
public:
    OptimizerNadam()
    {
		_fLearningRate =0.002f;
        beta1=0.9f;
        beta2=0.999f;
		beta1_prod = 0.f;
		beta2_prod = 0.f;
    }

    ~OptimizerNadam() override
    {}

	string name() const override
	{
		return "Nadam";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.002f;
		
		_m.resize(0,0);
        _v.resize(0,0);

        beta1_prod=beta1;
        beta2_prod=beta2;
    }

    virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
    {
        // init _m and _v if needed
        if (_v.size() == 0)
        {

            _m.setZero(dw.rows(), dw.cols());
            _v.setZero(dw.rows(), dw.cols());
        }

        // Nadam
        //m = beta1*m + (1-beta1)*dx
        //v = beta2*v + (1-beta2)*(dx**2)
        // w += (_m*beta1+dw*(1-beta1)/(1-beta1_prod) )*(-fLearningRate)/(sqrt(_v+epsilon));
        //beta1_prod*=beta1;
        //beta2_prod*=beta2;
		//alpha == learning_grate

        _m=_m*beta1+dw*(1.f-beta1);
        _v=_v*beta2+dw.cwiseAbs2()*(1.f-beta2);
        w += (_m*beta1+dw*(1.f-beta1)/(1.f-beta1_prod)  ).cwiseQuotient(_v.cwiseSqrt().cwiseMax(1.e-8f))*(-_fLearningRate);
        beta1_prod*=beta1;
        beta2_prod*=beta2;
    }
private:
    MatrixFloat _m, _v;
    float beta1, beta2, beta1_prod, beta2_prod;
};
//////////////////////////////////////////////////////////
// Adamax from http://ruder.io/optimizing-gradient-descent/index.html#adamax
class OptimizerAdamax : public Optimizer
{
public:
    OptimizerAdamax()
    {
		_fLearningRate =0.002f;
        beta1=0.9f;
        beta2=0.999f;
		beta1_prod = 0.f;
    }

    ~OptimizerAdamax() override
    {}

	string name() const override
	{
		return "Adamax";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.002f;

        _m.resize(0,0);
        _u.resize(0,0);

        beta1_prod=beta1;
    }

    virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
    {
        // init _m and _v if needed
        if (_u.size() == 0)
        {
            _m.setZero(dw.rows(), dw.cols());
            _u.setZero(dw.rows(), dw.cols());
        }

        //m = beta1*m + (1-beta1)*dw
        //u=max(b2*u,abs(dw))
        //W+=((-alpha)/(1-beta1_prod))*(m/(max(u,eps)))
        //beta1_prod*=beta1;
		//alpha == learning_grate

        _m=_m*beta1+dw*(1.f-beta1);
        _u=(_u*beta2).cwiseMax(dw.cwiseAbs());
        w += _m.cwiseQuotient(_u.cwiseMax(1.e-8f))*(-_fLearningRate/(1.f-beta1_prod));
        beta1_prod*=beta1;
    }
private:
    MatrixFloat _m, _u;
    float beta1, beta2, beta1_prod;
};
//////////////////////////////////////////////////////////
class OptimizerRPROPm : public Optimizer
{
public:
    OptimizerRPROPm()
    {}

    ~OptimizerRPROPm() override
    {}

	string name() const override
	{
		return "RPROP-";
	}

    virtual void init() override
    {
        _oldgradw.resize(0,0);
		_mu.resize(0,0);	
    }

    virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
    {
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

        // init if needed
        if (_oldgradw.size() == 0)
        {  
			_mu.resizeLike(dw);
			_mu.setOnes();
			_oldgradw=dw;
		}
		
		// update mu
/*
		sg=sign(gradw.*s.oldgradw);
		s.mu(sg<0) *= 0.5;
		s.mu(sg>0) *= 1.2;
*/ 	
		for(int i=0; i<_mu.size();i++)
		{
			float f=dw(i)*_oldgradw(i);

			if(f<0.f)
				_mu(i)*=0.5f;
			else if(f>0.f)
				_mu(i)*=1.2f;

			if (_mu(i) > 50.f)
				_mu(i) = 50.f;

			if (_mu(i) < 1.e-6f)
				_mu(i) = 1.e-6f;

			if (dw(i) > 0.f)
				w(i) -= _mu(i);
			else if (dw(i) < 0.f)
				w(i) += _mu(i);

			//     w.array()-= _mu.array().cwiseProduct(dw.array().cwiseSign()); // w=w-s.mu.*sign(gradw);

		
		}

   //     _mu = _mu.cwiseMax(1.e-6f); //s.mu(s.mu<1.e-6)=1.e-6;
   //     _mu = _mu.cwiseMin(50.f); //	s.mu(s.mu>50)=50;

		// update w
   //     w.array()-= _mu.array().cwiseProduct(dw.array().cwiseSign()); // w=w-s.mu.*sign(gradw);
		
		_oldgradw=dw;
    }
private:
    MatrixFloat _oldgradw,_mu;
};
//////////////////////////////////////////////////////////
Optimizer* create_optimizer(const string& sOptimizer)
{
    if (sOptimizer == "Adagrad")
        return new OptimizerAdagrad;

    if (sOptimizer == "Adam")
        return new OptimizerAdam;

    if (sOptimizer == "Adamax")
        return new OptimizerAdamax;

    if (sOptimizer == "Momentum")
        return new OptimizerMomentum;

    if (sOptimizer == "Nesterov")
        return new OptimizerNesterov;

    if (sOptimizer == "Nadam")
        return new OptimizerNadam;

    if (sOptimizer == "RMSProp")
        return new OptimizerRMSProp;

    if (sOptimizer == "RPROP-")
        return new OptimizerRPROPm;

    if (sOptimizer == "SGD")
        return new OptimizerSGD();

    return nullptr;
}
//////////////////////////////////////////////////////////////////////////////
void list_optimizers_available(vector<string>& vsOptimizers)
{
    vsOptimizers.clear();

    vsOptimizers.push_back("Adagrad");
    vsOptimizers.push_back("Adam");
    vsOptimizers.push_back("Nadam");
    vsOptimizers.push_back("Adamax");
    vsOptimizers.push_back("Momentum");
    vsOptimizers.push_back("Nesterov");
    vsOptimizers.push_back("RMSProp");
    vsOptimizers.push_back("RPROP-");
    vsOptimizers.push_back("SGD");
}
//////////////////////////////////////////////////////////////////////////////
