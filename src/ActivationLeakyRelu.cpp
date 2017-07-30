#include "ActivationLeakyRelu.h"

ActivationLeakyRelu::ActivationLeakyRelu(): Activation()
{ }

ActivationLeakyRelu::~ActivationLeakyRelu()
{ }

string ActivationLeakyRelu::name() const
{
    return "LeakyRelu";
}

double ActivationLeakyRelu::apply(double x) const
{
	return x>=0. ? x : 0.01*x;
}

double ActivationLeakyRelu::derivation(double x,double y) const
{
    (void)x;
    return y>=0. ? 1. : 0.01;
}
