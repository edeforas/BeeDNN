#include "ActivationRelu.h"

ActivationRelu::ActivationRelu(): Activation()
{ }

ActivationRelu::~ActivationRelu()
{ }

string ActivationRelu::name() const
{
    return "relu";
}

double ActivationRelu::forward(double x) const
{
	return x>=0 ? x : 0;	
}

double ActivationRelu::backward(double x,double y) const
{
    (void)x;
    return y>=0 ? 1 : 0; //f'(x) computed with y=f(x)
}
