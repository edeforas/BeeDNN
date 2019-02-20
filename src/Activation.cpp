#include "Activation.h"
#include <cmath>

// Activations functions
// the activation API use only the input to compute derivation because:
// -in minibatch derivation() is called sparsely
// -activation are not linear, we cannot use mean(data_out)=apply(mean(data_in)) (unless proven)
// -simplify the API

Activation::Activation()
{ }

Activation::~Activation()
{ }

//////////////////////////////////////////////////////////////////////////////
class ActivationAtan: public Activation
{
public:
    string name() const override
    {
        return "Atan";
    }

    float apply(float x) const override
    {
        return atanf(x);
    }

    float derivation(float x) const override
    {
        return 1.f/(1.f+x*x);
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationAsinh: public Activation
{
public:
    string name() const override
    {
        return "Asinh";
    }

    float apply(float x) const override
    {
        return asinhf(x);
    }

    float derivation(float x) const override
    {
        return 1.f/sqrtf(1.f+x*x); //http://mathworld.wolfram.com/InverseHyperbolicSine.html
    }
};//////////////////////////////////////////////////////////////////////////////
class ActivationElliot: public Activation
{
public:
    string name() const override
    {
        return "Elliot";
    }

    float apply(float x) const override
    {
        return 0.5f*(x/(1.f+fabs(x)))+0.5f;
    }

    float derivation(float x) const override
    {
        float d=1.f+fabs(x);
        return 0.5f/(d*d);
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationGauss: public Activation
{
public:
    string name() const override
    {
        return "Gauss";
    }

    float apply(float x) const override
    {
        return expf(-x*x);
    }

    float derivation(float x) const override
    {
        return -2.f*x*expf(-x*x);
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationLinear: public Activation
{
public:
    string name() const override
    {
        return "Linear";
    }

    float apply(float x) const override
    {
        return x;
    }

    float derivation(float x) const override
    {
        (void)x;
        return 1.f;
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationRelu: public Activation
{
public:
    string name() const override
    {
        return "Relu";
    }

    float apply(float x) const override
    {
        return x>0.f ? x : 0.f;
    }

    float derivation(float x) const override
    {
        return x>0.f ? 1.f : 0.f;
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationLeakyRelu: public Activation
{
public:
    string name() const override
    {
        return "LeakyRelu";
    }

    float apply(float x) const override
    {
        return x>=0.f ? x : 0.01f*x;
    }

    float derivation(float x) const override
    {
        return x>=0.f ? 1.f : 0.01f;
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationElu: public Activation
{
public:
    string name() const override
    {
        return "Elu";
    }

    float apply(float x) const override
    {
        if(x>=0.f)
            return x;
        else
            return expm1f(x);
    }

    float derivation(float x) const override
    {
        (void)x;

        if(x>=0.f)
            return 1.f;
        else
            return expm1f(x)+1.f;
    }
};
//////////////////////////////////////////////////////////////////////////////
#define SELU_LAMBDA (1.05070f)
#define SELU_ALPHA (1.67326f)
class ActivationSelu: public Activation
{
public:
    string name() const override
    {
        return "Selu";
    }

    float apply(float x) const override
    {
        if(x>=0.f)
            return SELU_LAMBDA*x;
        else
            return SELU_LAMBDA*SELU_ALPHA*expm1f(x);
    }

    float derivation(float x) const override
    {
        if(x>=0.f)
            return SELU_LAMBDA;
        else
            return SELU_LAMBDA*SELU_ALPHA*expm1f(x)+SELU_LAMBDA*SELU_ALPHA;
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationSoftPlus: public Activation
{
public:
    string name() const override
    {
        return "SoftPlus";
    }

    float apply(float x) const override
    {
        return log1pf(expf(x));
    }

    float derivation(float x) const override
    {
        return 1.f/(1.f+expf(-x)); //todo optimize
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationSoftSign: public Activation
{
public: 
    string name() const override
    {
        return "SoftSign";
    }
 
    float apply(float x) const override
    {
        return x/(1.f+fabsf(x));
    }

    float derivation(float x) const override
    {
        float d=1.f+fabsf(x);
        return 1.f/(d*d); //todo optimize
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationSigmoid: public Activation
{
public:

    string name() const override
    {
        return "Sigmoid";
    }

    float apply(float x) const override
    {
        return 1.f/(1.f+expf(-x));
    }
    float derivation(float x) const override
    {
        float s=1.f/(1.f+expf(-x));
        return s*(1.f-s); //todo optimise
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationTanh: public Activation
{
public:
    string name() const override
    {
        return "Tanh";
    }

    float apply(float x) const override
    {
        return tanhf(x);
    }
    float derivation(float x) const override
    {
        float t=tanhf(x);
        return 1.f-t*t;
    }
};
//////////////////////////////////////////////////////////////////////////////
//Swish as in the paper: Swish: A Self-Gated Activation Function
class ActivationSwish: public Activation
{
public:
    string name() const override
    {
        return "Swish";
    }

    float apply(float x) const override
    {
        return x/(1.f+expf(-x));
    }
    float derivation(float x) const override
    {
		float s=1.f/(1.f+expf(-x));
		return s*(x+1-x*s); //todo optimize
    }
};
//////////////////////////////////////////////////////////////////////////////
// Parablu is a softplus approximation without transendental function, author is Etienne de Foras
class ActivationParablu: public Activation
{
public:
    string name() const override
    {
        return "Parablu";
    }

    float apply(float x) const override
    {
        if(x<0.f)
            return 0.f;

        if(x>0.5f)
            return x-0.25f;

        return x*x;
    }
    float derivation(float x) const override
    {
        if(x<0.f)
            return 0.f;

        if(x>0.5f)
            return 1.f;

        return x+x;//2.*x
    }
};

//////////////////////////////////////////////////////////////////////////////
Activation* get_activation(const string& sActivation)
{
    if(sActivation=="Tanh")
        return new ActivationTanh;

    if(sActivation=="Asinh")
        return new ActivationAsinh;

    if(sActivation=="Sigmoid")
        return new ActivationSigmoid;

    if(sActivation=="Swish")
        return new ActivationSwish;

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

    return nullptr;
}
//////////////////////////////////////////////////////////////////////////////
void list_activations_available(vector<string>& vsActivations)
{
    vsActivations.clear();

    vsActivations.push_back("Tanh");
    vsActivations.push_back("TanhP1M2"); //not under testDNN
    vsActivations.push_back("Asinh");
    vsActivations.push_back("Sigmoid");
    vsActivations.push_back("Swish"); //not under tiny-dnn
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
    vsActivations.push_back("SoftMax"); //not under testDNN
    vsActivations.push_back("Parablu"); //not under tiny-dnn
}
//////////////////////////////////////////////////////////////////////////////
