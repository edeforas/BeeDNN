#ifndef NNEngine_
#define NNEngine_

#include "Matrix.h"

class NNEngine
{
public:
    NNEngine();
    virtual ~NNEngine();

    virtual void predict(const MatrixFloat pIn, MatrixFloat pOut)=0;
};

#endif
