/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// Bilinear as in : https://kikaben.com/swiglu-2020/

#include "LayerBilinear.h"

///////////////////////////////////////////////////////////////////////////////
LayerBilinear::LayerBilinear() :
    LayerGatedActivation("Identity", "Identity")
{ }
///////////////////////////////////////////////////////////////////////////////