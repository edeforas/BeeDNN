/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Activation_
#define Activation_

#include <string>
#include <vector>
using namespace std;

// Activations functions
// the activation API use only the input to compute derivation because:
// -in minibatch, derivation() is called sparsely
// -activation are not linear, we cannot use mean(data_out)=apply(mean(data_in)) (unless proven)
// -simplify the API

class Activation
{
public:
    Activation();
    virtual ~Activation();
    virtual string name() const =0;

    virtual float apply(float x) const =0;
    virtual float derivation(float x) const =0;
};

Activation* get_activation(const string & sActivation);
void list_activations_available(vector<string>& vsActivations);

#endif
