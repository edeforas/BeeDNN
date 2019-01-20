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

    virtual float apply(float x) const =0;
    virtual float derivation(float x) const =0;
};

Activation* get_activation(string sActivation);
void list_activations_available(vector<string>& vsActivations);

#endif
