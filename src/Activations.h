/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include <string>
#include <vector>

namespace beednn {
class Activation
{
public:
    Activation();
    virtual ~Activation();
    virtual std::string name() const =0;

    virtual float apply(float x) const =0;
    virtual float derivation(float x) const =0;
};

Activation* get_activation(const std::string & sActivation);
void list_activations_available(std::vector<std::string>& vsActivations);
}
