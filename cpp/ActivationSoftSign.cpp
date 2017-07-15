#include "ActivationSoftSign.h"
#include <cmath>

ActivationSoftSign::ActivationSoftSign(): Activation()
{ }

ActivationSoftSign::~ActivationSoftSign()
{ }

string ActivationSoftSign::name() const
{
    return "SoftSign";
}

double ActivationSoftSign::apply(double x) const
{
    return x/(1.+fabs(x));
}

double ActivationSoftSign::derivation(double x,double y) const
{
    (void)y;
    return 1./((1.+fabs(x))*(1.+fabs(x))); //todo optimize
}
