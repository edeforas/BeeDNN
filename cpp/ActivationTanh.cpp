#include "ActivationTanh.h"
#include <cmath>

ActivationTanh::ActivationTanh(): Activation()
{ }

ActivationTanh::~ActivationTanh()
{ }

string ActivationTanh::name() const
{
    return "tanh";
}

double ActivationTanh::forward(double x) const
{
    return tanh(x);
}

double ActivationTanh::backward(double x,double y) const
{
    (void)x;
    return 1.-y*y; //optimisation of f'(x) using y=f(x) in case of tanh
}
