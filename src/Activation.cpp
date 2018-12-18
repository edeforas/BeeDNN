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
        return 1.f/(1+x*x);
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
        return 0.5f*(x/(1.f+fabs(x)))+0.5f;
    }

    float derivation(float x,float y) const
    {
        (void)y;
        return 0.5f/((1.f+fabs(x))*(1.f+fabs(x))); //todo optimize
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
        return expf(-x*x);
    }

    float derivation(float x,float y) const
    {
        return -2.f*x*y; //derivate using f(x)
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
        return x>=0.f ? x : 0.f;
    }

    float derivation(float x,float y) const
    {
        (void)x;
        return y>=0.f ? 1.f : 0.f;
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
        return x>=0.f ? x : 0.01f*x;
    }

    float derivation(float x,float y) const
    {
        (void)x;
        return y>=0.f ? 1.f : 0.01f;
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

        if(y>=0.f)
            return 1.f;
        else
            return y+1.f; //optimisation of f'(x) using y=f(x) in case of Elu
    }
};
//////////////////////////////////////////////////////////////////////////////
#define SELU_LAMBDA 1.05070f
#define SELU_ALPHA 1.67326f
class ActivationSelu: public Activation
{
public:
    string name() const
    {
        return "Selu";
    }

    float apply(float x) const
    {
        if(x>=0.f)
            return SELU_LAMBDA*x;
        else
            return SELU_LAMBDA*SELU_ALPHA*expm1(x);
    }

    float derivation(float x,float y) const
    {
        (void)x;

        if(y>=0.f)
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
        return 1.f/(1.f+exp(-x));
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
        return x/(1.f+fabs(x));
    }

    float derivation(float x,float y) const
    {
        (void)y;
        return 1.f/((1.f+fabs(x))*(1.f+fabs(x))); //todo optimize
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
        return 1.f/(1.f+expf(-x));
    }
    float derivation(float x,float y) const
    {
        (void)x;
        return y*(1.f-y); //optimisation of f'(x) using y=f(x) in case of sigmoid
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
        return 1.f-y*y; //optimisation of f'(x) using y=f(x) in case of tanh
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
        if(x<0.f)
            return 0.f;

        if(x>0.5f)
            return x-0.25f;

        return x*x;
    }
    float derivation(float x,float y) const
    {
        (void)y;
        if(x<0.f)
            return 0.f;

        if(x>0.5f)
            return 1.f;

        return x+x;//2.*x
    }
};

//////////////////////////////////////////////////////////////////////////////
Activation* get_activation(string sActivation)
{
    if(sActivation=="Tanh")
        return new ActivationTanh;

    if(sActivation=="Sigmoid")
        return new ActivationSigmoid;

    if(sActivation=="Relu")
        return new ActivationRelu;

    if(sActivation=="Linear")
        return new ActivationLinear;

    if(sActivation=="Atan")
        return new ActivationAtan;

    if(sActivation=="Elliot")
        return new ActivationElliot;

    if(sActivation=="Gauss")
        return new ActivationGauss;

    if(sActivation=="LeakyRelu")
        return new ActivationLeakyRelu;

    if(sActivation=="Elu")
        return new ActivationElu;

    if(sActivation=="Selu")
        return new ActivationSelu;

    if(sActivation=="SoftPlus")
        return new ActivationSoftPlus;

    if(sActivation=="SoftSign")
        return new ActivationSoftSign;

    if(sActivation=="Parablu")
        return new ActivationParablu;

    return 0;
}
//////////////////////////////////////////////////////////////////////////////
void list_activations_available(vector<string>& vsActivations)
{
    vsActivations.clear();

    vsActivations.push_back("Tanh");
    vsActivations.push_back("Sigmoid");
    vsActivations.push_back("Relu");
    vsActivations.push_back("Linear");
    vsActivations.push_back("Atan");
    vsActivations.push_back("Elliot");
    vsActivations.push_back("Gauss");
    vsActivations.push_back("LeakyRelu");
    vsActivations.push_back("Elu");
    vsActivations.push_back("Selu");
    vsActivations.push_back("SoftPlus");
    vsActivations.push_back("SoftSign");
    vsActivations.push_back("Parablu");
}
//////////////////////////////////////////////////////////////////////////////
