#include "ActivationSoftPlus.h"
#include <cmath>

ActivationSoftPlus::ActivationSoftPlus(): Activation()
{ }

ActivationSoftPlus::~ActivationSoftPlus()
{ }

string ActivationSoftPlus::name() const
{
    return "SoftPlus";
}

double ActivationSoftPlus::apply(double x) const
{
    return log1p(exp(x));
}

double ActivationSoftPlus::derivation(double x,double y) const
{
    (void)y;
    return 1./(1.+exp(-x));
}
