#include "ActivationGauss.h"
#include <cmath>

ActivationGauss::ActivationGauss(): Activation()
{ }

ActivationGauss::~ActivationGauss()
{ }

string ActivationGauss::name() const
{
    return "Gauss";
}

double ActivationGauss::apply(double x) const
{
	return exp(-x*x);
}

double ActivationGauss::derivation(double x,double y) const
{
    return -2.*x*y; //derivate using f(x)
}
