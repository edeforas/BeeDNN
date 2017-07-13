#include "ActivationSelu.h"
#include <cmath>

#define SELU_LAMBDA 1.05070
#define SELU_ALPHA 1.67326

ActivationSelu::ActivationSelu(): Activation()
{ }

ActivationSelu::~ActivationSelu()
{ }

string ActivationSelu::name() const
{
    return "selu";
}

double ActivationSelu::apply(double x) const
{
	if(x>=0.)
		return SELU_LAMBDA*x;
	else
		return SELU_LAMBDA*SELU_ALPHA*(exp(x)-1.);
}

double ActivationSelu::derivation(double x,double y) const
{  
    (void)x;

	if(y>=0.)
		return SELU_LAMBDA;
	else
		return y+SELU_LAMBDA*SELU_ALPHA; //optimisation of f'(x) using y=f(x) in case of selu

}
