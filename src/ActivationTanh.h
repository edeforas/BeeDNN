#ifndef ActivationTanh_
#define ActivationTanh_

#include "Activation.h"

class ActivationTanh : public Activation
{
public:
    ActivationTanh();
    virtual ~ActivationTanh();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
