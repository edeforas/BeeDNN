#ifndef ActivationRelu_
#define ActivationRelu_

#include "Activation.h"

class ActivationRelu : public Activation
{
public:
    ActivationRelu();
    virtual ~ActivationRelu();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
