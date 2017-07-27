#include "ActivationAtan.h"
#include <cmath>

ActivationAtan::ActivationAtan(): Activation()
{ }

ActivationAtan::~ActivationAtan()
{ }

string ActivationAtan::name() const
{
    return "Atan";
}

double ActivationAtan::apply(double x) const
{
	return atan(x);	
}

double ActivationAtan::derivation(double x,double y) const
{
    (void)y;
    return 1./(1+x*x);
}
