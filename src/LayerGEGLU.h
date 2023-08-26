/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerGEGLU_
#define LayerGEGLU_

#include "LayerGatedActivation.h"
#include "Matrix.h"

namespace beednn {
class LayerGEGLU : public LayerGatedActivation
{
public:
    explicit LayerGEGLU();
};
}
#endif
