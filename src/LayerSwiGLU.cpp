/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// SwiGLU as in : https://kikaben.com/swiglu-2020/

#include "LayerSwiGLU.h"
namespace bee {

///////////////////////////////////////////////////////////////////////////////
LayerSwiGLU::LayerSwiGLU() :
    LayerGatedActivation("Identity", "Swish")
{ }
///////////////////////////////////////////////////////////////////////////////
}