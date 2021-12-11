/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include <cmath> // for sqrt
#include "Initializers.h"

// from https://www.tensorflow.org/api_docs/python/tf/keras/initializers

///////////////////////////////////////////////////////////////////////////////
void Initializers::GlorotUniform(MatrixFloat &m,Index iInputSize,Index iOutputSize)
{
    //Xavier Glorot uniform
    float a =sqrtf(6.f/(iInputSize + iOutputSize));
    m.setRandom(iInputSize, iOutputSize);
    m*=a;
}
///////////////////////////////////////////////////////////////////////////////
void Initializers::HeUniform(MatrixFloat& m, Index iInputSize, Index iOutputSize)
{
    //He uniform
    float a = sqrtf(6.f / iInputSize);
    m.setRandom(iInputSize, iOutputSize);
    m *= a;
}
///////////////////////////////////////////////////////////////////////////////
void Initializers::LecunUniform(MatrixFloat& m, Index iInputSize, Index iOutputSize)
{
    //He uniform
    float a = sqrtf(3.f / iInputSize);
    m.setRandom(iInputSize, iOutputSize);
    m *= a;
}
///////////////////////////////////////////////////////////////////////////////
void Initializers::Zeros(MatrixFloat &m,Index iInputSize,Index iOutputSize)
{
    //Zero initialization, used mainly on biases
    m.setZero(iInputSize, iOutputSize);
}
///////////////////////////////////////////////////////////////////////////////
void Initializers::Ones(MatrixFloat& m, Index iInputSize, Index iOutputSize)
{
    m.setOnes(iInputSize, iOutputSize);
}
///////////////////////////////////////////////////////////////////////////////