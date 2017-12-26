#include "Activation.h"
#include <cmath>

Activation::Activation()
{ }

Activation::~Activation()
{ }

//////////////////////////////////////////////////////////////////////////////
class ActivationAtan: public Activation
{
public:
    string name() const
    {
        return "Atan";
    }

    float apply(float x) const
    {
        return atan(x);
    }

    float derivation(float x,float y) const
    {
        (void)y;
        return 1./(1+x*x);
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationElliot: public Activation
{
public:
    string name() const
    {
        return "Elliot";
    }

    float apply(float x) const
    {
        return 0.5*(x/(1.+fabs(x)))+0.5;
    }

    float derivation(float x,float y) const
    {
        (void)y;
        return 0.5/((1.+fabs(x))*(1.+fabs(x))); //todo optimize
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationGauss: public Activation
{
public:
    string name() const
    {
        return "Gauss";
    }

    float apply(float x) const
    {
        return exp(-x*x);
    }

    float derivation(float x,float y) const
    {
        return -2.*x*y; //derivate using f(x)
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationLinear: public Activation
{
public:
    string name() const
    {
        return "Linear";
    }

    float apply(float x) const
    {
        return x;
    }

    float derivation(float x,float y) const
    {
        (void)x;
        (void)y;
        return 1.;
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationRelu: public Activation
{
public:
    string name() const
    {
        return "Relu";
    }

    float apply(float x) const
    {
        return x>=0. ? x : 0.;
    }

    float derivation(float x,float y) const
    {
        (void)x;
        return y>=0. ? 1. : 0.;
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationLeakyRelu: public Activation
{
public:
    string name() const
    {
        return "LeakyRelu";
    }

    float apply(float x) const
    {
        return x>=0. ? x : 0.01*x;
    }

    float derivation(float x,float y) const
    {
        (void)x;
        return y>=0. ? 1. : 0.01;
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationElu: public Activation
{
public:
    string name() const
    {
        return "Elu";
    }

    float apply(float x) const
    {
        if(x>=0.)
            return x;
        else
            return expm1(x);
    }

    float derivation(float x,float y) const
    {
        (void)x;

        if(y>=0.)
            return 1.;
        else
            return y+1.; //optimisation of f'(x) using y=f(x) in case of Elu
    }
};
//////////////////////////////////////////////////////////////////////////////
#define SELU_LAMBDA 1.05070
#define SELU_ALPHA 1.67326
class ActivationSelu: public Activation
{
public:
    string name() const
    {
        return "Selu";
    }

    float apply(float x) const
    {
        if(x>=0.)
            return SELU_LAMBDA*x;
        else
            return SELU_LAMBDA*SELU_ALPHA*expm1(x);
    }

    float derivation(float x,float y) const
    {
        (void)x;

        if(y>=0.)
            return SELU_LAMBDA;
        else
            return y+SELU_LAMBDA*SELU_ALPHA; //optimisation of f'(x) using y=f(x) in case of selu
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationSoftPlus: public Activation
{
public:
    string name() const
    {
        return "SoftPlus";
    }

    float apply(float x) const
    {
        return log1p(exp(x));
    }

    float derivation(float x,float y) const
    {
        (void)y;
        return 1./(1.+exp(-x));
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationSoftSign: public Activation
{
public: 
    string name() const
    {
        return "SoftSign";
    }

    float apply(float x) const
    {
        return x/(1.+fabs(x));
    }

    float derivation(float x,float y) const
    {
        (void)y;
        return 1./((1.+fabs(x))*(1.+fabs(x))); //todo optimize
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationSigmoid: public Activation
{
public:

    string name() const
    {
        return "Sigmoid";
    }

    float apply(float x) const
    {
        return 1./(1.+exp(-x));
    }
    float derivation(float x,float y) const
    {
        (void)x;
        return y*(1.-y); //optimisation of f'(x) using y=f(x) in case of sigmoid
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationTanh: public Activation
{
public:
    string name() const
    {
        return "Tanh";
    }

    float apply(float x) const
    {
        return tanh(x);
    }
    float derivation(float x,float y) const
    {
        (void)x;
        return 1.-y*y; //optimisation of f'(x) using y=f(x) in case of tanh
    }
};



//////////////////////////////////////////////////////////////////////////////
class ActivationParablu: public Activation
{
public:
    string name() const
    {
        return "Parablu";
    }

    float apply(float x) const
    {
		if(x<0.)
			return 0;
		
        if(x>2.)
            return x-1.;
		
        return x*x/4.;
    }
    float derivation(float x,float y) const
    {
        (void)y;
		if(x<0.)
			return 0.;
		
        if(x>2.)
			return 1.;
		
        return x/2.;
    }
};
//////////////////////////////////////////////////////////////////////////////
ActivationManager::ActivationManager()
{
    _vActivations.push_back(new ActivationAtan);
    _vActivations.push_back(new ActivationElliot);
    _vActivations.push_back(new ActivationGauss);
    _vActivations.push_back(new ActivationLinear);
    _vActivations.push_back(new ActivationRelu);
    _vActivations.push_back(new ActivationLeakyRelu);
    _vActivations.push_back(new ActivationElu);
    _vActivations.push_back(new ActivationSelu);
    _vActivations.push_back(new ActivationSoftPlus);
    _vActivations.push_back(new ActivationSoftSign);
    _vActivations.push_back(new ActivationSigmoid);
    _vActivations.push_back(new ActivationTanh);
    _vActivations.push_back(new ActivationParablu);
	}
//////////////////////////////////////////////////////////////////////////////
ActivationManager::~ActivationManager()
{
    for(unsigned int i=0; i<_vActivations.size();i++)
        delete _vActivations[i];
}
//////////////////////////////////////////////////////////////////////////////
Activation* ActivationManager::get_activation(const string& sName) //do not delete: manager own it.
{
    for(unsigned int i=0; i<_vActivations.size();i++) //todo use map
    {
        if(_vActivations[i]->name()==sName)
            return _vActivations[i];
    }

    return 0;
}
//////////////////////////////////////////////////////////////////////////////
void ActivationManager::list_all(vector<string>& allActivationNames) const
{
    allActivationNames.clear();
    for(unsigned int i=0; i<_vActivations.size();i++)
        allActivationNames.push_back(_vActivations[i]->name());
}
//////////////////////////////////////////////////////////////////////////////
