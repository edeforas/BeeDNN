/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include <cmath> // for sqrt
#include "Initializers.h"
#include <iostream>
// from https://www.tensorflow.org/api_docs/python/tf/keras/initializers
namespace bee{

///////////////////////////////////////////////////////////////////////////////
void Initializers::GlorotUniform(MatrixFloat &m,Index iInputSize,Index iOutputSize)
{
    //Xavier Glorot uniform
    float a =sqrtf(6.f/(iInputSize + iOutputSize));
    m.setRandom(iInputSize, iOutputSize);
    m*=a;
}
///////////////////////////////////////////////////////////////////////////////
void Initializers::GlorotNormal(MatrixFloat& m, Index iInputSize, Index iOutputSize)
{
    //Xavier Glorot normal
    float a = sqrtf(2.f / (iInputSize + iOutputSize));
    m.resize(iInputSize, iOutputSize);
    setRandomNormal(m,0.f,a);
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
void Initializers::HeNormal(MatrixFloat& m, Index iInputSize, Index iOutputSize)
{
    //He Normal
    float a = sqrtf(2.f / (iInputSize ));
    m.resize(iInputSize, iOutputSize);
    setRandomNormal(m, 0.f, a);
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
void Initializers::LecunNormal(MatrixFloat& m, Index iInputSize, Index iOutputSize)
{
    //He Normal
    float a = sqrtf(1.f / (iInputSize));
    m.resize(iInputSize, iOutputSize);
    setRandomNormal(m, 0.f, a);
}
///////////////////////////////////////////////////////////////////////////////
void Initializers::Uniform(MatrixFloat& m, Index iInputSize, Index iOutputSize)
{
    m.setRandom(iInputSize, iOutputSize);
}
///////////////////////////////////////////////////////////////////////////////
void Initializers::Normal(MatrixFloat& m, Index iInputSize, Index iOutputSize)
{
    m.resize(iInputSize, iOutputSize);
    setRandomNormal(m, 0.f, 1.f);
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
void Initializers::compute(const string& sInitializer, MatrixFloat& m, Index iInputSize, Index iOutputSize)
{
    // Todo , optimize, but not mandatory for now
    if (sInitializer == "GlorotUniform")
        Initializers::GlorotUniform(m, iInputSize, iOutputSize);

    else if (sInitializer == "GlorotNormal")
        Initializers::GlorotNormal(m, iInputSize, iOutputSize);

    else if (sInitializer == "HeUniform")
        Initializers::HeUniform(m, iInputSize, iOutputSize);

    else if (sInitializer == "HeNormal")
        Initializers::HeNormal(m, iInputSize, iOutputSize);

    else if (sInitializer == "LecunUniform")
        Initializers::LecunUniform(m, iInputSize, iOutputSize);

    else if (sInitializer == "LecunNormal")
        Initializers::LecunNormal(m, iInputSize, iOutputSize);

    else if (sInitializer == "Uniform")
        Initializers::Uniform(m, iInputSize, iOutputSize);

    else if (sInitializer == "Normal")
        Initializers::Normal(m, iInputSize, iOutputSize);

    else if (sInitializer == "Zeros")
        Initializers::Zeros(m, iInputSize, iOutputSize);

    else if (sInitializer == "Ones")
        Initializers::Ones(m, iInputSize, iOutputSize);

    // todo : throw an error if here
}
///////////////////////////////////////////////////////////////////////////////
}