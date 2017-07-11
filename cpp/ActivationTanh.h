#ifndef ActivationTanh_
#define ActivationTanh_

#include "Activation.h"

class ActivationTanh : public Activation
{
public:
    ActivationTanh();
    virtual ~ActivationTanh();
    virtual string name() const;

    virtual double forward(double x) const;
    virtual double backward(double x,double y) const;
};

#endif
