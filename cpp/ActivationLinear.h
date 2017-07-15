#ifndef ActivationLinear_
#define ActivationLinear_

#include "Activation.h"

class ActivationLinear : public Activation
{
public:
    ActivationLinear();
    virtual ~ActivationLinear();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
