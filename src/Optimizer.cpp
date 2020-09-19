/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Optimizer.h"

#include <cassert>
#include <cmath>

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
void Optimizer::set_learningrate(float fLearningRate)  //-1.f is for default params
{
	_fLearningRate = fLearningRate;
	init();
}
//////////////////////////////////////////////////////////
float Optimizer::get_learningrate()  //-1.f is for default params
{
	return _fLearningRate;
}
//////////////////////////////////////////////////////////
void Optimizer::set_params(float fLearningRate, float fDecay, float fMomentum)  //-1.f is for default params
{
	_fLearningRate = fLearningRate;
	_fDecay = fDecay;
	_fMomentum = fMomentum;

	init();
}
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
class OptimizerNone : public Optimizer // placeholder for no optimizer
{
public:
	OptimizerNone() :Optimizer()
	{}

	~OptimizerNone() override
	{}

	Optimizer* clone() override
	{
		return new OptimizerNone;
	}

	string name() const override
	{
		return "None";
	}

	virtual void init() override
	{ }

	virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
	{
		(void)w;
		(void)dw;
	}
};
//////////////////////////////////////////////////////////
class OptimizerStep : public Optimizer
{
public:	
    OptimizerStep():Optimizer()
    {}

    ~OptimizerStep() override
    {}
	
	Optimizer* clone() override
	{
		auto pOpt=new OptimizerStep;
		pOpt->_fLearningRate = _fLearningRate;
		return pOpt;
	}

	string name() const override
	{
		return "Step";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.1f;
	}

    virtual void optimize(MatrixFloat& w,const MatrixFloat& dw) override
    {
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

        // Vanilla step update
        //	x =x -learning_rate * gradient_direction
        w -=  dw.cwiseSign() * _fLearningRate ;
    }
};
//////////////////////////////////////////////////////////
class OptimizerSGD : public Optimizer
{
public:	
    OptimizerSGD() :Optimizer()
    {}

    ~OptimizerSGD() override
    {}
	
	Optimizer* clone() override
	{
		auto pOpt = new OptimizerSGD;
		pOpt->_fLearningRate = _fLearningRate;
		return pOpt;
	}

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
// Momentum from https://cs231n.github.io/neural-networks-3/#sgd
// simplified version of momentum, (avoid one mult)
class OptimizerMomentum : public Optimizer
{
public:
    OptimizerMomentum() :Optimizer()
    {}

    ~OptimizerMomentum() override
    {}

	Optimizer* clone() override
	{
		auto pOpt = new OptimizerMomentum;
		pOpt->_fLearningRate = _fLearningRate;
		pOpt->_fMomentum = _fMomentum;
		return pOpt;
	}

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
			_v.setZero(w.rows(),w.cols());

        _v = _v*_fMomentum + dw*_fLearningRate;
        w -= _v;
    }
private:
    MatrixFloat _v;
};

//////////////////////////////////////////////////////////
// Momentum from https://cs231n.github.io/neural-networks-3/#sgd
// Andrew Ng version
class OptimizerMomentumNg : public Optimizer
{
public:
    OptimizerMomentumNg() :Optimizer()
    {}

    ~OptimizerMomentumNg() override
    {}

	Optimizer* clone() override
	{
		auto pOpt = new OptimizerMomentumNg;
		pOpt->_fLearningRate = _fLearningRate;
		pOpt->_fMomentum = _fMomentum;
		return pOpt;
	}

	string name() const override
	{
		return "MomentumNg";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.1f; // lr can be high here
		if (_fMomentum == -1.f) _fMomentum = 0.9f;

        _v.resize(0,0);
    }

    virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
    {
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

        // init _V if needed
        if (_v.size() == 0)
            _v=dw;

        _v = _v*_fMomentum + dw*(1.f-_fMomentum); // recursive averaging
        w -= _v*_fLearningRate;
    }
private:
    MatrixFloat _v;
};

//////////////////////////////////////////////////////////
// Nesterov from https://cs231n.github.io/neural-networks-3/#sgd
class OptimizerNesterov : public Optimizer
{
public:
    OptimizerNesterov() :Optimizer()
    {}

    ~OptimizerNesterov() override
    {}

	Optimizer* clone() override
	{
		auto pOpt = new OptimizerNesterov;
		pOpt->_fLearningRate = _fLearningRate;
		pOpt->_fMomentum = _fMomentum;
		return pOpt;
	}

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

        _v_prev = _v;
        _v = _v*_fMomentum - dw*_fLearningRate ;
        w += -_v_prev*(_fMomentum) + _v*(1.f + _fMomentum) ;
    }
private:
    MatrixFloat _v, _v_prev;
};
//////////////////////////////////////////////////////////////////////////////
class OptimizerAdagrad : public Optimizer
{
public:
    OptimizerAdagrad() :Optimizer()
    {}

    ~OptimizerAdagrad() override
    {}

	Optimizer* clone() override
	{
		auto pOpt = new OptimizerAdagrad;
		pOpt->_fLearningRate = _fLearningRate;
		return pOpt;
	}

	string name() const override
	{
		return "Adagrad";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f)
			_fLearningRate = 0.001f;

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
    OptimizerRMSProp() :Optimizer()
    {}

    ~OptimizerRMSProp() override
    {}

	Optimizer* clone() override
	{
		auto pOpt = new OptimizerRMSProp;
		pOpt->_fLearningRate = _fLearningRate;
		pOpt->_fDecay = _fDecay;
		return pOpt;
	}

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
//from https ://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
class OptimizerAdam : public Optimizer
{
public:
    OptimizerAdam() :Optimizer()
    {
        beta1=0.9f;
        beta2=0.999f;
		beta1_prod = 0.f;
		beta2_prod = 0.f;
    }

    ~OptimizerAdam() override
    {}

	Optimizer* clone() override
	{
		auto pOpt = new OptimizerAdam;
		pOpt->_fLearningRate = _fLearningRate;
		return pOpt;
	}

	string name() const override
	{
		return "Adam";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f)
			_fLearningRate = 0.001f;

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
		float invBeta2 = 1.f / (1.f - beta2_prod);

        _m=_m*beta1+dw*(1.f-beta1);
        _v=_v*beta2+dw.cwiseAbs2()*(1.f-beta2);
        w -= _m.cwiseQuotient((_v*invBeta2).cwiseSqrt().cwiseMax(1.e-8f))*(_fLearningRate/(1.f-beta1_prod));

		beta1_prod*=beta1;
        beta2_prod*=beta2;
    }
private:
    MatrixFloat _m, _v;
    float beta1, beta2, beta1_prod, beta2_prod;
};
/////////////////////////////////////////////////////////////////////////////////////
class OptimizerAdamW : public Optimizer
{
public:
	OptimizerAdamW() :Optimizer()
	{
		beta1 = 0.9f;
		beta2 = 0.999f;
		beta1_prod = 0.f;
		beta2_prod = 0.f;
		lambda_regul = 0.000001f;
	}

	~OptimizerAdamW() override
	{}

	Optimizer* clone() override
	{
		auto pOpt = new OptimizerAdamW;
		pOpt->_fLearningRate = _fLearningRate;
		return pOpt;
	}

	string name() const override
	{
		return "AdamW";
	}

	virtual void init() override
	{
		if (_fLearningRate == -1.f)
			_fLearningRate = 0.001f;

		_m.resize(0, 0);
		_v.resize(0, 0);

		beta1_prod = beta1;
		beta2_prod = beta2;
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

		float invBeta2 = 1.f / (1.f - beta2_prod);

		_m = _m * beta1 + dw * (1.f - beta1);
		_v = _v * beta2 + dw.cwiseAbs2()*(1.f - beta2);
		w -= _m.cwiseQuotient((_v*invBeta2).cwiseSqrt().cwiseMax(1.e-8f))*(_fLearningRate / (1.f - beta1_prod))+w*lambda_regul;

		beta1_prod *= beta1;
		beta2_prod *= beta2;
	}
private:
	MatrixFloat _m, _v;
	float beta1, beta2, beta1_prod, beta2_prod, lambda_regul;
};
//////////////////////////////////////////////////////////
class OptimizerAmsgrad : public Optimizer
{
public:
	OptimizerAmsgrad() :Optimizer()
	{
		beta1 = 0.9f;
		beta2 = 0.999f;
	}

	~OptimizerAmsgrad() override
	{}

	Optimizer* clone() override
	{
		auto pOpt = new OptimizerAmsgrad;
		pOpt->_fLearningRate = _fLearningRate;
		return pOpt;
	}

	string name() const override
	{
		return "Amsgrad";
	}

	virtual void init() override
	{
		if (_fLearningRate == -1.f)
			_fLearningRate = 0.01f;

		_v.resize(0, 0);
	}

	virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
	{
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

		// init if needed
		if (_v.size() == 0)
		{
			_v.setZero(dw.rows(), dw.cols());
			_v_hat.setZero(dw.rows(), dw.cols());
			_m.setZero(dw.rows(), dw.cols());
		}

		_m = _m * beta1 + dw * (1.f - beta1);
		_v = _v * beta2 + dw.cwiseAbs2()*(1.f - beta2);
		_v_hat = _v.cwiseMax(_v_hat);

		w -= (_m.cwiseQuotient(_v_hat.cwiseSqrt().cwiseMax(1.e-8f)))*_fLearningRate;
	}
private:
	MatrixFloat _m, _v,_v_hat;
	float beta1, beta2;
};
/////////////////////////////////////////////////////////////////////////////////////
class OptimizerNadam : public Optimizer
{
public:
    OptimizerNadam() :Optimizer()
    {
		_fLearningRate =0.001f;
        beta1=0.9f;
        beta2=0.999f;
		beta1_prod = 0.f;
		beta2_prod = 0.f;
    }

    ~OptimizerNadam() override
    {}

	Optimizer* clone() override
	{
		auto pOpt = new OptimizerNadam;
		pOpt->_fLearningRate = _fLearningRate;
		return pOpt;
	}

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
    OptimizerAdamax() :Optimizer()
    {
		_fLearningRate =0.002f;
        beta1=0.9f;
        beta2=0.999f;
		beta1_prod = 0.f;
    }

    ~OptimizerAdamax() override
    {}

	Optimizer* clone() override
	{
		auto pOpt = new OptimizerAdamax;
		pOpt->_fLearningRate = _fLearningRate;
		return pOpt;
	}

	string name() const override
	{
		return "Adamax";
	}

    virtual void init() override
    {
		if (_fLearningRate == -1.f) _fLearningRate = 0.001f;

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
//RPROP-  as in : https://pdfs.semanticscholar.org/df9c/6a3843d54a28138a596acc85a96367a064c2.pdf
// or in paper : Improving the Rprop Learning Algorithm (Christian Igel and Michael Husken)
class OptimizerRPROPm : public Optimizer
{
public:
    OptimizerRPROPm() :Optimizer()
    {}

    ~OptimizerRPROPm() override
    {}

	Optimizer* clone() override
	{
		return new OptimizerRPROPm;
	}

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
			_mu.setConstant(0.0125f);
			_oldgradw=dw;
		}
		
		for(int i=0; i<_mu.size();i++)
		{
			float f=dw(i)*_oldgradw(i);
			float& mu=_mu(i);
			
			if(f<0.f)
			{	
				mu*=0.5f;
				if (mu < 1.e-6f)
					mu = 1.e-6f;
			}
			else if(f>0.f)
			{
				mu*=1.2f;
				if (mu > 50.f)
					mu = 50.f;
			}

			if (dw(i) > 0.f)
				w(i) -= mu;
			else if (dw(i) < 0.f)
				w(i) += mu;
		}

		_oldgradw=dw;
    }
private:
    MatrixFloat _oldgradw,_mu;
};
//////////////////////////////////////////////////////////
//iRPROP-  as in : https://pdfs.semanticscholar.org/df9c/6a3843d54a28138a596acc85a96367a064c2.pdf
// or in paper : Improving the Rprop Learning Algorithm (Christian Igel and Michael Husken)
class OptimizeriRPROPm : public Optimizer
{
public:
	OptimizeriRPROPm() :Optimizer()
	{}

	~OptimizeriRPROPm() override
	{}

	Optimizer* clone() override
	{
		return new OptimizerRPROPm;
	}

	string name() const override
	{
		return "iRPROP-";
	}

	virtual void init() override
	{
		_oldgradw.resize(0, 0);
		_mu.resize(0, 0);
	}

	virtual void optimize(MatrixFloat& w, const MatrixFloat& dw) override
	{
		assert(w.rows() == dw.rows());
		assert(w.cols() == dw.cols());

		// init if needed
		if (_oldgradw.size() == 0)
		{
			_mu.resizeLike(dw);
			_mu.setConstant(0.0125f);
			_oldgradw = dw;
		}

		for (int i = 0; i < _mu.size(); i++)
		{
			float dwi = dw(i);
			float f = dwi *_oldgradw(i);
			float& mu = _mu(i);

			if (f < 0.f)
			{
				mu *= 0.5f;
				if (mu < 1.e-6f)
					mu = 1.e-6f;

				dwi = 0;
			}
			else if (f > 0.f)
			{
				mu *= 1.2f;
				if (mu > 50.f)
					mu = 50.f;
			}

			if (dwi > 0.f)
				w(i) -= mu;
			else if (dwi < 0.f)
				w(i) += mu;
		
			_oldgradw(i) = dwi;
		}
	}
private:
	MatrixFloat _oldgradw, _mu;
};

//////////////////////////////////////////////////////////
Optimizer* create_optimizer(const string& sOptimizer)
{
	if (sOptimizer == "None")
		return new OptimizerNone;
	
	if (sOptimizer == "Adagrad")
        return new OptimizerAdagrad;

    if (sOptimizer == "Adam")
        return new OptimizerAdam;

	if (sOptimizer == "AdamW")
		return new OptimizerAdamW;

	if (sOptimizer == "Amsgrad")
		return new OptimizerAmsgrad;

	if (sOptimizer == "Adamax")
        return new OptimizerAdamax;

    if (sOptimizer == "Momentum")
        return new OptimizerMomentum;

	if (sOptimizer == "MomentumNg")
		return new OptimizerMomentumNg;
	
	if (sOptimizer == "Nesterov")
        return new OptimizerNesterov;

    if (sOptimizer == "Nadam")
        return new OptimizerNadam;

    if (sOptimizer == "RMSProp")
        return new OptimizerRMSProp;

    if (sOptimizer == "RPROP-")
        return new OptimizerRPROPm;

    if (sOptimizer == "iRPROP-")
        return new OptimizeriRPROPm;
	
    if (sOptimizer == "SGD")
        return new OptimizerSGD();

    if (sOptimizer == "Step")
        return new OptimizerStep();

    return nullptr;
}
//////////////////////////////////////////////////////////////////////////////
void list_optimizers_available(vector<string>& vsOptimizers)
{
    vsOptimizers.clear();

	vsOptimizers.push_back("None");
	vsOptimizers.push_back("Adagrad");
    vsOptimizers.push_back("Adam");
	vsOptimizers.push_back("AdamW");
	vsOptimizers.push_back("Amsgrad");
	vsOptimizers.push_back("Nadam");
    vsOptimizers.push_back("Adamax");
    vsOptimizers.push_back("Momentum");
	vsOptimizers.push_back("MomentumNg");
	vsOptimizers.push_back("Nesterov");
    vsOptimizers.push_back("RMSProp");
    vsOptimizers.push_back("RPROP-");
	vsOptimizers.push_back("iRPROP-");
	vsOptimizers.push_back("SGD");
	vsOptimizers.push_back("Step");
}
//////////////////////////////////////////////////////////////////////////////
