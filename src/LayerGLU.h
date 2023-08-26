/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef LayerGLU_
#define LayerGLU_

#include "LayerGatedActivation.h"
namespace beednn {
class LayerGLU : public LayerGatedActivation
{
public:
	explicit LayerGLU();
};
}
#endif
