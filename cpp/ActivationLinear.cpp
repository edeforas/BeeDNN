#include "ActivationLinear.h"

ActivationLinear::ActivationLinear(): Activation()
{ }

ActivationLinear::~ActivationLinear()
{ }

string ActivationLinear::name() const
{
    return "linear";
}

double ActivationLinear::apply(double x) const
{
	return x;	
}

double ActivationLinear::derivation(double x,double y) const
{
    (void)x;
    (void)y;
    return 1;
}
