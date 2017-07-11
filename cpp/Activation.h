#ifndef Activation_
#define Activation_

#include <string>
using namespace std;

class Activation
{
public:
    Activation();
    virtual ~Activation();
    virtual string name() const =0;

    virtual double forward(double x) const =0;
    virtual double backward(double x,double y) const =0;
};

#endif
