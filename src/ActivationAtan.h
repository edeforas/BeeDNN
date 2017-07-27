#ifndef ActivationAtan_
#define ActivationAtan_

#include "Activation.h"

class ActivationAtan : public Activation
{
public:
    ActivationAtan();
    virtual ~ActivationAtan();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
