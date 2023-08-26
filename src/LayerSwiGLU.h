/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerSwiGLU_
#define LayerSwiGLU_

#include "LayerGatedActivation.h"
namespace beednn {
class LayerSwiGLU : public LayerGatedActivation
{
public:
    explicit LayerSwiGLU();
};
}
#endif
