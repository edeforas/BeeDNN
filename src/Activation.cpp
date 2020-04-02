/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "Activation.h"
#include <cmath>

Activation::Activation()
{ }

Activation::~Activation()
{ }

//////////////////////////////////////////////////////////////////////////////
// as in : https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons
class ActivationAbsolute: public Activation
{
public:
    string name() const override
    {
        return "Absolute";
    }

    float apply(float x) const override
    {
        return fabs(x);
    }

    float derivation(float x) const override
    {
        return (x>=0.f)?1.f:-1.f;
    }
};
//////////////////////////////////////////////////////////////////////////////
// as in : http://mathworld.wolfram.com/InverseHyperbolicSine.html
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
        return 1.f/sqrtf(1.f+x*x);
    }
};
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
// Bipolar as in: https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons
class ActivationBipolar: public Activation
{
public:
	string name() const override
	{
		return "Bipolar";
	}

	float apply(float x) const override
	{
		if (x > 0.f)
			return 1.f;
		else
			return -1.f;
	}
	float derivation(float x) const override
	{
		(void)x;

		return 0.f;
	}
};
//////////////////////////////////////////////////////////////////////////////
// BipolarSigmoid as in: https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
class ActivationBipolarSigmoid: public Activation
{
public:
    string name() const override
    {
        return "BipolarSigmoid";
    }

    float apply(float x) const override
    {
		float s=expf(x);
        return (s-1.f)/(s+1.f);
	}
    float derivation(float x) const override
    {
		float s=expf(x);
        return 2.f*s/((s+1.f)*(s+1.f)); //todo optimise
    }
};
//////////////////////////////////////////////////////////////////////////////
// ComplementaryLogLog as in: https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons
class ActivationComplementaryLogLog : public Activation
{
public:
	string name() const override
	{
		return "ComplementaryLogLog";
	}

	float apply(float x) const override
	{
		return 1.f - expf(-expf(x));
	}
	float derivation(float x) const override
	{
		return expf(x-expf(x)); 
	}
};
//////////////////////////////////////////////////////////////////////////////
class ActivationDivideBy256 : public Activation
{
public:
	string name() const override
	{
		return "DivideBy256";
	}

	float apply(float x) const override
	{
		return x * 0.00390625f; //in fix, use (x+127) >> 8;
	}

	float derivation(float x) const override
	{
        (void)x;
		return 0.00390625f; //1.f/256.f
	}
};
//////////////////////////////////////////////////////////////////////////////
// ELiSH as in: https://arxiv.org/pdf/1808.00783.pdf
// or paper: The Quest for the Golden Activation Function ; Mina Basirat and Peter M. Roth
class ActivationELiSH: public Activation
{
public:
    string name() const override
    {
        return "ELiSH";
    }

    float apply(float x) const override
    {
        float exm=expf(-x);
        if(x>=0.f)
            return x/(1.f+exm);
		else
            return (1.f/exm-1.f)/(1.f+exm);
	}
    float derivation(float x) const override
    {
		if(x>=0.f)
		{
			float exm=expf(-x);
            float invex=1.f/(exm+1.f);
			return invex+x*exm*invex*invex;
		}
		else
		{
            float exm=expf(-x);
            float invex=1.f/(exm+1.f);
            float ex=1.f/exm;
            return ex*invex+exm*(ex-1.f)*invex*invex;
		}
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
class ActivationExponential: public Activation
{
public:
    string name() const override
    {
        return "Exponential";
    }

    float apply(float x) const override
    {
        return expf(x);
    }

    float derivation(float x) const override
    {
        return expf(x);
    }
};
//////////////////////////////////////////////////////////////////////////////
// E2RU as in https://arxiv.org/pdf/1804.11237.pdf
class ActivationE2RU : public Activation
{
public:
	string name() const override
	{
		return "E2RU";
	}

	float apply(float x) const override
	{
		if (x >= 0.f)
			return sqrtf(2.f*2.f*x+1.f)-0.5f;
		else
			return expf(x*2.f)-0.5f;
	}

	float derivation(float x) const override
	{
		if (x >= 0.f)
			return 2.f/sqrtf(2.f*2.f*x+1.f);
		else
			return 2.f*expf(2.f*x);
	}
};
//////////////////////////////////////////////////////////////////////////////
// E3RU as in https://arxiv.org/pdf/1804.11237.pdf
class ActivationE3RU : public Activation
{
public:
	string name() const override
	{
		return "E3RU";
	}

	float apply(float x) const override
	{
		if (x >= 0.f)
			return powf(3.f*3.f*x + 1.f,1.f/3.f) - 1.f/3.f;
		else
			return expf(x*3.f) - 1.f/3.f;
	}

	float derivation(float x) const override
	{
		if (x >= 0.f)
			return 3.f*powf(3.f*3.f*x + 1.f,-2.f/3.f);
		else
			return 3.f*expf(3.f*x);
	}
};
//////////////////////////////////////////////////////////////////////////////
// FTS as in the paper: https://arxiv.org/ftp/arxiv/papers/1812/1812.06247.pdf
// with best constant T=-0.2
class ActivationFTS : public Activation
{
public:
	string name() const override
	{
		return "FTS";
	}

	float apply(float x) const override
	{
		if (x < 0.f)
			return -0.2f;
		else return x / (1.f + expf(-x))-0.2f;
	}
	float derivation(float x) const override
	{
		if (x < 0.f)
			return 0.f;

		float s = 1.f / (1.f + expf(-x));
		return s * (x + 1.f - x * s);
	}
};
//////////////////////////////////////////////////////////////////////////////
// FTS+ as in the website: https://medium.com/@lessw/comparison-of-activation-functions-for-deep-learning-initial-winner-ftswish-f13e2621847
// with best constant T=-0.25
class ActivationFTSPlus : public Activation
{
public:
	string name() const override
	{
		return "FTS+";
	}

	float apply(float x) const override
	{
		if (x < 0.f)
			return -0.25f;
		else return x / (1.f + expf(-x)) - 0.25f;
	}
	float derivation(float x) const override
	{
		if (x < 0.f)
			return 0.f;

		float s = 1.f / (1.f + expf(-x));
		return s * (x + 1.f - x * s);
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
// Binary Step as in: https://en.wikipedia.org/wiki/Activation_function
class ActivationBinaryStep : public Activation
{
public:
	string name() const override
	{
		return "BinaryStep";
	}

	float apply(float x) const override
	{
		if (x > 0.f)
			return 1.f;
		else
			return 0.f;
	}
	float derivation(float x) const override
	{
		(void)x;

		return 0.f;
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
// GELU from the paper: https://arxiv.org/pdf/1606.08415.pdf
// or GAUSSIAN ERROR LINEAR UNITS (GELUS) ; Dan Hendrycks and Kevin Gimpel
// and https://medium.com/@shoray.goel/gelu-gaussian-error-linear-unit-4ec59fb2e47c
// approximation version and numbers rounded to 1.e-6
class ActivationGELU: public Activation
{
public:
    string name() const override
    {
        return "GELU";
    }

    float apply(float x) const override
    {
        return 0.5f*x*(1.f+tanhf(1.772454f*x*(1.f+0.044715f*x*x)));
    }

    float derivation(float x) const override
    {
        float fSech=1.f/coshf(1.772454f*x*(0.044715f*x*x+1.f));
        return 0.5f*tanh(1.772454f*x*(0.044715f*x*x+1.f))+x*(0.118883f*x*x+0.886227f)*fSech*fSech+0.5f;
    }
};
/////////////////////////////////////////////////////////////////////////////
// HardELU, ELU approximation, quick and easy to convert in fixed point
// Author is Minh Tri LE
class ActivationHardELU: public Activation
{
public:
    string name() const override
    {
        return "HardELU";
    }

    float apply(float x) const override
    {
        if(x>=0.f)
            return x;

        if(x<=-2.f)
            return -1.f;

        return x*0.5f;
    }

    float derivation(float x) const override
    {
        if(x>=0.f)
            return 1.f;

        if(x<=-2.f)
            return 0.f;

        return 0.5f;
    }
};
//////////////////////////////////////////////////////////////////////////////
//HardShrink from https://nn.readthedocs.io/en/rtd/transfer/
// default lambda is 0.5
class ActivationHardShrink: public Activation
{
public:
    string name() const override
    {
        return "HardShrink";
    }

    float apply(float x) const override
    {
        if(x>0.5f)
            return x;

        if(x<-0.5f)
            return x;

        return 0.f;
    }
    float derivation(float x) const override
    {
        if(x>0.5f)
            return 1.f;

        if(x<-0.5f)
            return 1.f;

        return 0.f;
    }
};
//////////////////////////////////////////////////////////////////////////////
//HardTanh from https://cs224d.stanford.edu/lecture_notes/LectureNotes3.pdf
class ActivationHardTanh: public Activation
{
public:
    string name() const override
    {
        return "HardTanh";
    }

    float apply(float x) const override
    {
        if(x>1.f)
            return 1.f;

        if(x<-1.f)
            return -1.f;

        return x;
    }
    float derivation(float x) const override
    {
        if(x>1.f)
            return 0.f;

        if(x<-1.f)
            return 0.f;

        return 1.f;
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationIdentity : public Activation
{
public:
	string name() const override
	{
		return "Identity";
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
//from : https://arxiv.org/pdf/1710.09967.pdf
#define ISRLU_ALPHA (1.0f)
class ActivationISRLU: public Activation
{
public:
    string name() const override
    {
        return "ISRLU";
    }

    float apply(float x) const override
    {
        if(x>=0.f)
            return x;
        else
            return x/sqrtf(1.f+ ISRLU_ALPHA*x*x);
	}

    float derivation(float x) const override
    {
        if(x>=0.f)
            return 1.f;
        else
		{
			float t=1.f/sqrtf(1.f+ ISRLU_ALPHA*x*x);
			return t*t*t;
		}
	}
};
//////////////////////////////////////////////////////////////////////////////
// Mish from : https://arxiv.org/abs/1908.08681
// and https://github.com/digantamisra98/Mish
// or paper: Mish: A Self Regularized Non-Monotonic Neural Activation Function ; Diganta Misra
class ActivationMish: public Activation
{
public:
    string name() const override
    {
        return "Mish";
    }
    float apply(float x) const override
    {
		float tempSoftplus=log1pf(expf(x));
        return x*tanhf(tempSoftplus);
    }
    float derivation(float x) const override
    {
        //version from derivative computation
        float ex=expf(x);
        float tempSoftplus=log1pf(ex);
        float tempSech=1.f/coshf(tempSoftplus);
        return tanh(tempSoftplus)+  x*ex*tempSech * tempSech/(ex+1.f);
    }
};
//////////////////////////////////////////////////////////////////////////////
// from: https://en.wikipedia.org/wiki/Activation_function
class ActivationSin : public Activation
{
public:
	string name() const override
	{
		return "Sin";
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
//from: https://en.wikipedia.org/wiki/Activation_function
class ActivationSinC : public Activation
{
public:
	string name() const override
	{
		return "SinC";
	}

	float apply(float x) const override
	{
		if (x == 0.f)
			return 1.f;

		return sinf(x) / x;
	}

	float derivation(float x) const override
	{
		if (x == 0.f)
			return 0.f;

		return cosf(x) / x - sinf(x) / (x*x);
	}
};
//////////////////////////////////////////////////////////////////////////////
// Logit as in : https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
class ActivationLogit : public Activation
{
public:
    string name() const override
    {
        return "Logit";
    }

    float apply(float x) const override
    {
        return logf(x / (1.f -x));
    }
    float derivation(float x) const override
    {
        return -x/(x-1.f);
    }
};
//////////////////////////////////////////////////////////////////////////////
// LogSigmoid as in : https://nn.readthedocs.io/en/rtd/transfer/
class ActivationLogSigmoid : public Activation
{
public:
	string name() const override
	{
		return "LogSigmoid";
	}

	float apply(float x) const override
	{
		return logf(1.f / (1.f + expf(-x)));
	}
	float derivation(float x) const override
	{
		return 1.f / (1.f + expf(x));
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
class ActivationRelu6: public Activation
{
public:
    string name() const override
    {
        return "Relu6";
    }

    float apply(float x) const override
    {
        if(x>=6.f)
            return 6.f;

        if(x<=0.f)
            return 0.f;

        return x;
    }

    float derivation(float x) const override
    {
        if(x>=6.f)
            return 0.f;

        if(x<=0.f)
            return 0.f;

        return 1.f;
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
// LeakyRelu compatible with integer computation (using a shift >> 8 ) . author: Etienne de Foras
class ActivationLeakyRelu256 : public Activation
{
public:
    string name() const override
    {
        return "LeakyRelu256";
    }

    float apply(float x) const override
    {
        return x >= 0.f ? x : x* 0.00390625f;  // 1/256 ; will be (x+127)>>8 in fixed point
    }

    float derivation(float x) const override
    {
        return x >= 0.f ? 1.f : 0.00390625f; // 1/256 ; will be (x+127)>>8 in fixed point
    }
};
//////////////////////////////////////////////////////////////////////////////
// from https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
// and https://tensorlayer.readthedocs.io/en/latest/modules/activation.html#tensorlayer.activation.leaky_twice_relu6
// for now, alpha=alpha_high=0.2f ; for IoT, use a power of two, i.e 0.125 or 0.25
class ActivationLeakyTwiceRelu6: public Activation
{
public:
    string name() const override
    {
        return "LeakyTwiceRelu6";
    }

    float apply(float x) const override
    {
        if(x<0.f)
            return 0.2f*x;

        if(x<6.f)
            return x;

        return 6.f+0.2f*(x-6.f); //can be optimized
    }

    float derivation(float x) const override
    {
        if(x<0.f)
            return 0.2f;

        if(x>6.f)
            return 0.2f;

        return 1.f;
    }
};
//////////////////////////////////////////////////////////////////////////////
// Lecun's Tanh as in : https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
class ActivationLecunTanh : public Activation
{
public:
	string name() const override
	{
		return "LecunTanh";
	}

	float apply(float x) const override
	{
		return 1.7159f*tanhf(x*0.666666f);
	}
	float derivation(float x) const override
	{
		float t = coshf(x*0.666666f);
		return (1.7159f*0.666666f) / (t*t);
	}
};
//////////////////////////////////////////////////////////////////////////////
// LiSHT as in https://arxiv.org/pdf/1901.05894.pdf
class ActivationLiSHT : public Activation
{
public:
	string name() const override
	{
		return "LiSHT";
	}

	float apply(float x) const override
	{
		return x*tanhf(x);
	}
	float derivation(float x) const override
	{
		float t = tanhf(x);
		return x+t*(1.f-x*t);
	}
};
//////////////////////////////////////////////////////////////////////////////
class ActivationNLRelu: public Activation
{
public:
    string name() const override
    {
        return "NLRelu";
    }

    float apply(float x) const override
    {
        return x>0.f ? log1pf(x) : 0.f;
    }

    float derivation(float x) const override
    {
        return x>0.f ? 1.f/(1.f+x) : 0.f;
    }
};
//////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////
// O2RU as in https://arxiv.org/pdf/1804.11237.pdf
class ActivationO2RU : public Activation
{
public:
	string name() const override
	{
		return "O2RU";
	}

	float apply(float x) const override
	{
		if (x >= 0.f)
			return sqrtf(2.f*2.f*x + 1.f) - 1.f;
		else
			return -sqrtf(-2.f*2.f*x + 1.f) + 1.f;
	}

	float derivation(float x) const override
	{
		if (x >= 0.f)
			return 2.f / sqrtf(2.f*2.f*x + 1.f);
		else
			return 2.f / sqrtf(-2.f*2.f*x + 1.f);
	}
};
//////////////////////////////////////////////////////////////////////////////
// O3RU as in https://arxiv.org/pdf/1804.11237.pdf
class ActivationO3RU : public Activation
{
public:
	string name() const override
	{
		return "O3RU";
	}

	float apply(float x) const override
	{
		if (x >= 0.f)
			return powf(3.f*3.f*x + 1.f, 1.f / 3.f) - 1.f;
		else
			return -powf(-3.f*3.f*x + 1.f, 1.f / 3.f) + 1.f;
	}

	float derivation(float x) const override
	{
		if (x >= 0.f)
			return 3.f*powf(3.f*3.f*x + 1.f, -2.f / 3.f);
		else
			return 3.f*powf(-3.f*3.f*x + 1.f, -2.f / 3.f);
	}
};
//////////////////////////////////////////////////////////////////////////////


class ActivationELU: public Activation
{
public:
    string name() const override
    {
        return "ELU";
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
//HardShrink from https://www.gabormelli.com/RKB/Softshrink_Activation_Function
// default lambda is 0.5
class ActivationSoftShrink: public Activation
{
public:
    string name() const override
    {
        return "SoftShrink";
    }

    float apply(float x) const override
    {
        if(x>0.5f)
            return x-0.5f;

        if(x<-0.5f)
            return x+0.5f;

        return 0.f;
    }
    float derivation(float x) const override
    {
        if(x>0.5f)
            return 1.f;

        if(x<-0.5f)
            return 1.f;

        return 0.f;
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
// from : https://arxiv.org/pdf/1702.03118.pdf
// or paper : Sigmoid-Weighted Linear Units for Neural Network Function ; Stefan Elfwinga Eiji Uchibea Kenji Doyab
class ActivationSiLU: public Activation
{
public:
    string name() const override
    {
        return "SiLU";
    }
    float apply(float x) const override
    {
        return x/(1.f+expf(-x));
    }
    float derivation(float x) const override
    {
        float ex=expf(-x);
        float exinv=1.f/(1.f+ex);
        return exinv*(1.f+x*ex*exinv);
    }
};
//////////////////////////////////////////////////////////////////////////////
// from : https://arxiv.org/pdf/1702.03118.pdf
// or paper : Sigmoid-Weighted Linear Units for Neural Network Function ; Stefan Elfwinga Eiji Uchibea Kenji Doyab
class ActivationdSiLU: public Activation
{
public:
    string name() const override
    {
        return "dSiLU";
    }
    float apply(float x) const override
    {
        float ex=expf(-x);
        float exinv=1.f/(1.f+ex);
        return exinv*(1.f+x*ex*exinv);
    }
    float derivation(float x) const override
    {
        float ex=expf(-x);
        float exinv=1.f/(1.f+ex);

//		return -x*ex*exinv*exinv+2.f*ex*exinv*exinv+2.f*x*ex*ex*exinv*exinv*exinv;
		return ex*exinv*exinv*(2.f+x*(2.f*ex*exinv-1.f));
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
            return 0.f;

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
        return s*(x+1.f-x*s);
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
// Parablu is a softplus approximation without slow elementary function, author is Etienne de Foras
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

        return x+x; //2.f*x
    }
};
//////////////////////////////////////////////////////////////////////////////
// SQ-RBF as in : Computationally Efficient Radial Basis Function ; Adedamola Wuraola, Nitish D. Patel
class ActivationSQRBF: public Activation
{
public:
    string name() const override
    {
        return "SQ-RBF";
    }

    float apply(float x) const override
    {
        if(fabs(x)>=2.f)
            return 0.f;

		if(fabs(x)<=1.f)
			return 1.f-x*x*0.5f;
		
        if(x>=0.f)
            return x*x*0.5f-2.f*x+2.f; //todo factorize
        else
            return x*x*0.5f+2.f*x+2.f; //todo factorize
    }
    float derivation(float x) const override
    {
        if(fabs(x)>=2.f)
            return 0.f;

		if(fabs(x)<=1.f)
			return 1.f-x;

        if(x>=0.f)
            return x-2.f;
        else
            return x+2.f;
    }
};
//////////////////////////////////////////////////////////////////////////////
class ActivationTanh : public Activation
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
		float t = tanhf(x);
		return 1.f - t * t; //same as 1/square(cosh(x))
	}
};
//////////////////////////////////////////////////////////////////////////////
class ActivationTanhShrink : public Activation
{
public:
	string name() const override
	{
		return "TanhShrink";
	}

	float apply(float x) const override
	{
		return x-tanhf(x);
	}
	float derivation(float x) const override
	{
		float t = tanhf(x);
		return  t * t;
	}
};
//////////////////////////////////////////////////////////////////////////////
// as in: https://keras.io/layers/advanced-activations/
// for now with theta =1
class ActivationThresholdedRelu : public Activation
{
public:
	string name() const override
	{
        return "ThresholdedRelu";
	}

	float apply(float x) const override
	{
		return x >= 1.f ? x : 0.f;
	}

	float derivation(float x) const override
	{
		return x >= 1.f ? 1.f : 0.f;
	}
};
//////////////////////////////////////////////////////////////////////////////
Activation* get_activation(const string& sActivation)
{
    if(sActivation=="Absolute")
        return new ActivationAbsolute;

    else if(sActivation=="Asinh")
        return new ActivationAsinh;

    else if(sActivation=="Atan")
        return new ActivationAtan;

    else if(sActivation=="Bent")
        return new ActivationBent;

	else if(sActivation == "BinaryStep")
		return new ActivationBinaryStep;

	else if(sActivation == "Bipolar")
		return new ActivationBipolar;

	else if(sActivation == "BipolarSigmoid")
		return new ActivationBipolarSigmoid;

	else if (sActivation == "ComplementaryLogLog")
		return new ActivationComplementaryLogLog;

	else if(sActivation == "DivideBy256")
		return new ActivationDivideBy256;

	else if(sActivation == "dSiLU")
		return new ActivationdSiLU;

    else if(sActivation=="ELiSH")
        return new ActivationELiSH;

    else if(sActivation=="Elliot")
        return new ActivationElliot;

    else if(sActivation=="ELU")
        return new ActivationELU;

    else if(sActivation=="Exponential")
        return new ActivationExponential;

	else if (sActivation == "E2RU")
		return new ActivationE2RU;
	
	else if (sActivation == "E3RU")
		return new ActivationE3RU;

	else if (sActivation == "FTS")
		return new ActivationFTS;

	else if (sActivation == "FTS+")
		return new ActivationFTSPlus;

	else if(sActivation=="Gauss")
        return new ActivationGauss;
	
    else if(sActivation=="GELU")
        return new ActivationGELU;

    else if(sActivation=="HardELU")
        return new ActivationHardELU;

    else if(sActivation=="HardSigmoid")
        return new ActivationHardSigmoid;

    else if(sActivation=="HardShrink")
        return new ActivationHardShrink;

    else if(sActivation=="HardTanh")
        return new ActivationHardTanh;

    else if(sActivation=="Identity")
        return new ActivationIdentity;

    else if(sActivation=="ISRLU")
        return new ActivationISRLU;
	
    else if(sActivation=="LeakyRelu")
        return new ActivationLeakyRelu;

    else if(sActivation=="LeakyRelu256")
        return new ActivationLeakyRelu256;

    else if(sActivation=="LeakyTwiceRelu6")
        return new ActivationLeakyTwiceRelu6;

	else if (sActivation == "LecunTanh")
		return new ActivationLecunTanh;
	
	else if (sActivation == "LiSHT")
		return new ActivationLiSHT;

	else if(sActivation == "Logit")
        return new ActivationLogit;

	else if(sActivation == "LogSigmoid")
		return new ActivationLogSigmoid;

	else if(sActivation=="Mish")
        return new ActivationMish;
	
	else if(sActivation=="NLRelu")
        return new ActivationNLRelu;

	else if (sActivation == "O2RU")
		return new ActivationO2RU;

	else if (sActivation == "O3RU")
		return new ActivationO3RU;

    else if(sActivation=="Parablu")
        return new ActivationParablu;

    else if(sActivation=="Relu")
        return new ActivationRelu;

    else if(sActivation=="Relu6")
        return new ActivationRelu6;

    else if(sActivation=="Selu")
        return new ActivationSelu;

    else if(sActivation=="SQNL")
        return new ActivationSQNL;

    else if(sActivation=="SQ-RBF")
        return new ActivationSQRBF;

    else if(sActivation=="SoftPlus")
        return new ActivationSoftPlus;

    else if(sActivation=="Sigmoid")
        return new ActivationSigmoid;

    else if(sActivation=="SiLU")
        return new ActivationSiLU;

    else if(sActivation=="Sin")
        return new ActivationSin;

    else if(sActivation=="SinC")
        return new ActivationSinC;

    else if(sActivation=="Swish")
        return new ActivationSwish;
    
	else if(sActivation=="SoftShrink")
        return new ActivationSoftShrink;
	
    else if(sActivation=="SoftSign")
        return new ActivationSoftSign;

    else if(sActivation=="Tanh")
        return new ActivationTanh;

    else if(sActivation=="TanhShrink")
        return new ActivationTanhShrink;

    else if (sActivation == "ThresholdedRelu")
		return new ActivationThresholdedRelu;

    return nullptr;
}
//////////////////////////////////////////////////////////////////////////////
void list_activations_available(vector<string>& vsActivations)
{
    vsActivations.clear();

    vsActivations.push_back("Absolute");
    vsActivations.push_back("Asinh");
    vsActivations.push_back("Atan");
    vsActivations.push_back("Bent");
	vsActivations.push_back("BinaryStep");
	vsActivations.push_back("Bipolar");
	vsActivations.push_back("BipolarSigmoid");
	vsActivations.push_back("ComplementaryLogLog");
	vsActivations.push_back("DivideBy256");
	vsActivations.push_back("dSiLU");
    vsActivations.push_back("ELiSH");
    vsActivations.push_back("Elliot");
    vsActivations.push_back("ELU");
	vsActivations.push_back("Exponential");
	vsActivations.push_back("E2RU");
	vsActivations.push_back("E3RU");
	vsActivations.push_back("FTS");
	vsActivations.push_back("FTS+");
	vsActivations.push_back("Gauss");
    vsActivations.push_back("GELU");
    vsActivations.push_back("HardELU");
    vsActivations.push_back("HardSigmoid");
    vsActivations.push_back("HardShrink");
    vsActivations.push_back("HardTanh");
    vsActivations.push_back("Identity");
	vsActivations.push_back("ISRLU");
    vsActivations.push_back("LeakyRelu");
    vsActivations.push_back("LeakyRelu256");
    vsActivations.push_back("LeakyTwiceRelu6");
	vsActivations.push_back("LecunTanh");
	vsActivations.push_back("LiSHT");
	vsActivations.push_back("Logit");
    vsActivations.push_back("LogSigmoid");
    vsActivations.push_back("Mish");	
    vsActivations.push_back("NLRelu");
	vsActivations.push_back("O2RU");
	vsActivations.push_back("O3RU");
    vsActivations.push_back("Parablu");
    vsActivations.push_back("Relu");
    vsActivations.push_back("Relu6");
    vsActivations.push_back("Selu");
    vsActivations.push_back("SoftPlus");
    vsActivations.push_back("SoftShrink");
    vsActivations.push_back("SoftSign");
    vsActivations.push_back("SQNL");
    vsActivations.push_back("SQ-RBF");
    vsActivations.push_back("Sigmoid");
    vsActivations.push_back("SiLU");
    vsActivations.push_back("SinC");
    vsActivations.push_back("Sin");
    vsActivations.push_back("Swish");
    vsActivations.push_back("Tanh");
	vsActivations.push_back("TanhShrink");
	vsActivations.push_back("ThresholdedRelu");
}
//////////////////////////////////////////////////////////////////////////////
