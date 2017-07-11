#include "ActivationSigmoid.h"
#include <cmath>

ActivationSigmoid::ActivationSigmoid(): Activation()
{ }

ActivationSigmoid::~ActivationSigmoid()
{ }

string ActivationSigmoid::name() const
{
    return "sigmoid";
}

double ActivationSigmoid::forward(double x) const
{
	return 1./(1.+exp(-x));	
}

double ActivationSigmoid::backward(double x,double y) const
{
    (void)x;
    return y*(1.-y); //optimisation of f'(x) using y=f(x) in case of sigmoid
}
