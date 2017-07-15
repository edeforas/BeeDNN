#ifndef ActivationElliot_
#define ActivationElliot_

#include "Activation.h"

class ActivationElliot : public Activation
{
public:
    ActivationElliot();
    virtual ~ActivationElliot();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
