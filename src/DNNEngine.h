#ifndef DNNEngine_
#define DNNEngine_

#include "Matrix.h"

class DNNEngine
{
public:
    DNNEngine();
    virtual ~DNNEngine();

    virtual void predict(const MatrixFloat mIn, MatrixFloat mOut)=0;
};

#endif
