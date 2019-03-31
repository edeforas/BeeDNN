/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

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
};
//////////////////////////////////////////////////////////////////////////////
class ActivationSin: public Activation
{
public:
    string name() const override
    {
        return "Sin"; //from: https://en.wikipedia.org/wiki/Activation_function
    }

    float apply(float x) const override
    {
        return sinf(x);
    }

    float derivation(float x) const override
    {
        return cosf(x);
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationSinC: public Activation
{
public:
    string name() const override
    {
        return "SinC"; //from: https://en.wikipedia.org/wiki/Activation_function
    }

    float apply(float x) const override
    {
        if(x==0.f)
            return 1.f;

        return sinf(x)/x;
    }

    float derivation(float x) const override
    {
        if(x==0.f)
            return 0.f;

        return cosf(x)/x-sinf(x)/(x*x);
    }
};
//////////////////////////////////////////////////////////////////////////////
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
class ActivationBent: public Activation
{
public:
    string name() const override
    {
        return "Bent";
    }

    float apply(float x) const override
    {
        return (sqrtf(x*x+1.f)-1.f)*0.5f+x;
    }

    float derivation(float x) const override
    {
        return x/(2.f*sqrtf(x*x+1.f))+1.f;
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
// hard sigmoid as in: https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L277
class ActivationHardSigmoid: public Activation
{
public:
    string name() const override
    {
        return "HardSigmoid";
    }

    float apply(float x) const override
    {
        if(x>2.5f)
            return 1.f;

        if(x<-2.5f)
            return -1.f;

        return 0.2f*x+0.5f;
    }
    float derivation(float x) const override
    {
        if(x>2.5f)
            return 0.f;

        if(x<-2.5f)
            return 0.f;

        return 0.2f;
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
class ActivationSQNL: public Activation  //from: https://en.wikipedia.org/wiki/Activation_function
{
public:
    string name() const override
    {
        return "SQNL";
    }

    float apply(float x) const override
    {
        if(x>=0.f)
        {
            if(x>=2.0f)
                return 1.f;

            return x-x*x*0.25f;
        }
        else
        {
            if(x<=-2.0f)
                return -1.f;

            return x+x*x*0.25f;
        }
    }
    float derivation(float x) const override
    {
        if(x>=0.f)
        {
            if(x>=2.0f)
                return 0.f;

            return 1.f-x*0.5f;
        }
        else
        {
            if(x<=-2.0f)
                return 0.f;

            return 1.f+x*0.5f;
        }
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

        return x+x;//2.f*x
    }
};
//////////////////////////////////////////////////////////////////////////////
Activation* get_activation(const string& sActivation)
{
    if(sActivation=="Asinh")
        return new ActivationAsinh;

    if(sActivation=="Atan")
        return new ActivationAtan;

    if(sActivation=="Bent")
        return new ActivationBent;

    if(sActivation=="Elliot")
        return new ActivationElliot;

    if(sActivation=="Elu")
        return new ActivationElu;

    if(sActivation=="HardSigmoid")
        return new ActivationHardSigmoid;

    if(sActivation=="Gauss")
        return new ActivationGauss;

    if(sActivation=="Linear")
        return new ActivationLinear;

    if(sActivation=="LeakyRelu")
        return new ActivationLeakyRelu;

    if(sActivation=="Parablu")
        return new ActivationParablu;

    if(sActivation=="Relu")
        return new ActivationRelu;

    if(sActivation=="Selu")
        return new ActivationSelu;

    if(sActivation=="SQNL")
        return new ActivationSQNL;

    if(sActivation=="SoftPlus")
        return new ActivationSoftPlus;

    if(sActivation=="Sin")
        return new ActivationSin;

    if(sActivation=="SinC")
        return new ActivationSinC;

    if(sActivation=="Sigmoid")
        return new ActivationSigmoid;

    if(sActivation=="Swish")
        return new ActivationSwish;

    if(sActivation=="SoftSign")
        return new ActivationSoftSign;

    if(sActivation=="Tanh")
        return new ActivationTanh;

    return nullptr;
}
//////////////////////////////////////////////////////////////////////////////
void list_activations_available(vector<string>& vsActivations)
{
    vsActivations.clear();

    vsActivations.push_back("Asinh");
    vsActivations.push_back("Atan");
    vsActivations.push_back("Bent"); //not under tiny-dnn
    vsActivations.push_back("Elliot");
    vsActivations.push_back("Elu");
    vsActivations.push_back("Gauss");
    vsActivations.push_back("HardSigmoid");
    vsActivations.push_back("Linear");
    vsActivations.push_back("LeakyRelu");
    vsActivations.push_back("Parablu"); //not under tiny-dnn
    vsActivations.push_back("Relu");
    vsActivations.push_back("Selu");
    vsActivations.push_back("SoftPlus");
    vsActivations.push_back("SoftSign");
    vsActivations.push_back("SQNL");  //not under tiny-dnn
    vsActivations.push_back("Sigmoid");
    vsActivations.push_back("SinC");  //not under tiny-dnn
    vsActivations.push_back("Sin");
    vsActivations.push_back("Swish"); //not under tiny-dnn
    vsActivations.push_back("Tanh");
}
//////////////////////////////////////////////////////////////////////////////
