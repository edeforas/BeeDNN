#ifndef ActivationSoftPlus_
#define ActivationSoftPlus_

#include "Activation.h"

class ActivationSoftPlus : public Activation
{
public:
    ActivationSoftPlus();
    virtual ~ActivationSoftPlus();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
