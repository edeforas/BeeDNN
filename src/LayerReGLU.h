/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerReGLU_
#define LayerReGLU_

#include "LayerGatedActivation.h"
namespace bee {
class LayerReGLU : public LayerGatedActivation
{
public:
    explicit LayerReGLU();
};
}
#endif
