#ifndef ActivationElu_
#define ActivationElu_

#include "Activation.h"

class ActivationElu : public Activation
{
public:
    ActivationElu();
    virtual ~ActivationElu();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
