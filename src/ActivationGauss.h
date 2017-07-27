#ifndef ActivationGauss_
#define ActivationGauss_

#include "Activation.h"

class ActivationGauss : public Activation
{
public:
    ActivationGauss();
    virtual ~ActivationGauss();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
