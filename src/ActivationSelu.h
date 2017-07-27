#ifndef ActivationSelu_
#define ActivationSelu_

#include "Activation.h"

class ActivationSelu : public Activation
{
public:
    ActivationSelu();
    virtual ~ActivationSelu();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
