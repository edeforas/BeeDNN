#include "ActivationElu.h"
#include <cmath>

ActivationElu::ActivationElu(): Activation()
{ }

ActivationElu::~ActivationElu()
{ }

string ActivationElu::name() const
{
    return "Elu";
}

double ActivationElu::apply(double x) const
{
	if(x>=0.)
		return x;
	else
		return expm1(x);
}

double ActivationElu::derivation(double x,double y) const
{  
    (void)x;

	if(y>=0.)
		return 1.;
	else
		return y+1.; //optimisation of f'(x) using y=f(x) in case of Elu
}
