/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include <cmath> // for sqrt
#include "Initializers.h"

///////////////////////////////////////////////////////////////////////////////
void Initializers::XavierUniform(MatrixFloat &m,Index iInputSize,Index iOutputSize)
{
    //Xavier uniform initialization
    float a =sqrtf(6.f/(iInputSize + iOutputSize));
    m.setRandom(iInputSize, iOutputSize);
    m*=a;
}
///////////////////////////////////////////////////////////////////////////////
void Initializers::Zero(MatrixFloat &m,Index iInputSize,Index iOutputSize)
{
    //Zero initialization, used mainly on biases
    m.setZero(iInputSize, iOutputSize);
}
///////////////////////////////////////////////////////////////////////////////
