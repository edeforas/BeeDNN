#ifndef ActivationLeakyRelu_
#define ActivationLeakyRelu_

#include "Activation.h"

class ActivationLeakyRelu : public Activation
{
public:
    ActivationLeakyRelu();
    virtual ~ActivationLeakyRelu();
    virtual string name() const;

    virtual double apply(double x) const;
    virtual double derivation(double x,double y) const;
};

#endif
