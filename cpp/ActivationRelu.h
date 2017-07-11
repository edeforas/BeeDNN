#ifndef ActivationRelu_
#define ActivationRelu_

#include "Activation.h"

class ActivationRelu : public Activation
{
public:
    ActivationRelu();
    virtual ~ActivationRelu();
    virtual string name() const;

    virtual double forward(double x) const;
    virtual double backward(double x,double y) const;
};

#endif
