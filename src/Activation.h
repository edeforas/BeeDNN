#ifndef Activation_
#define Activation_

#include <string>
#include <vector>
using namespace std;

class Activation
{
public:
    Activation();
    virtual ~Activation();
    virtual string name() const =0;

    virtual double apply(double x) const =0;
    virtual double derivation(double x,double y) const =0;
};

class ActivationManager
{
public:
    ActivationManager();
    virtual ~ActivationManager();

    Activation* get_activation(const string& sName); //do not delete: manager own it.

    void list_all(vector<string>& allActivationNames) const;

private:
    vector<Activation*> _vActivations;
};

#endif
