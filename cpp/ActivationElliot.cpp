#include "ActivationElliot.h"
#include <cmath>

ActivationElliot::ActivationElliot(): Activation()
{ }

ActivationElliot::~ActivationElliot()
{ }

string ActivationElliot::name() const
{
    return "Elliot";
}

double ActivationElliot::apply(double x) const
{
	return 0.5*(x/(1.+fabs(x)))+0.5;
}

double ActivationElliot::derivation(double x,double y) const
{
    (void)y;
    return 0.5/((1.+fabs(x))*(1.+fabs(x))); //todo optimize
}
