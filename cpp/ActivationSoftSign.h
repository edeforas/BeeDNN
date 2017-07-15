#ifndef ActivationSoftSign_
#define ActivationSoftSign_

#include "Activation.h"

class ActivationSoftSign : public Activation
{
public:
    ActivationSoftSign();
    virtual ~ActivationSoftSign();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
