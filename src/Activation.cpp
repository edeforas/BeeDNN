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

    double apply(double x) const
    {
        return atan(x);
    }

    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        return 0.5*(x/(1.+fabs(x)))+0.5;
    }

    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        return exp(-x*x);
    }

    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        return x;
    }

    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        return x>=0. ? x : 0.;
    }

    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        return x>=0. ? x : 0.01*x;
    }

    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        if(x>=0.)
            return x;
        else
            return expm1(x);
    }

    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        if(x>=0.)
            return SELU_LAMBDA*x;
        else
            return SELU_LAMBDA*SELU_ALPHA*expm1(x);
    }

    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        return log1p(exp(x));
    }

    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        return x/(1.+fabs(x));
    }

    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        return 1./(1.+exp(-x));
    }
    double derivation(double x,double y) const
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

    double apply(double x) const
    {
        return tanh(x);
    }
    double derivation(double x,double y) const
    {
        (void)x;
        return 1.-y*y; //optimisation of f'(x) using y=f(x) in case of tanh
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
}

ActivationManager::~ActivationManager()
{
    for(unsigned int i=0; i<_vActivations.size();i++)
        delete _vActivations[i];
}

Activation* ActivationManager::get_activation(const string& sName) //do not delete: manager own it.
{
    for(unsigned int i=0; i<_vActivations.size();i++) //todo use map
    {
        if(_vActivations[i]->name()==sName)
            return _vActivations[i];
    }

    return 0;
}

void ActivationManager::list_all(vector<string>& allActivationNames)
{
    allActivationNames.clear();
    for(unsigned int i=0; i<_vActivations.size();i++)
        allActivationNames.push_back(_vActivations[i]->name());
}
