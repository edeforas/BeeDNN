#ifndef ActivationSigmoid_
#define ActivationSigmoid_

#include "Activation.h"

class ActivationSigmoid : public Activation
{
public:
    ActivationSigmoid();
    virtual ~ActivationSigmoid();
    virtual string name() const;

    virtual double forward(double x) const;
    virtual double backward(double x,double y) const;
};

#endif
