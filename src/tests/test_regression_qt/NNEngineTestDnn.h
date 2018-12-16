#ifndef NNEngineTestDnn_
#define NNEngineTestDnn_

#include "NNEngine.h"

class Net;

class NNEngineTestDnn : public NNEngine
{
public:
    NNEngineTestDnn();
    virtual ~NNEngineTestDnn();

    virtual void predict(const MatrixFloat pIn, MatrixFloat pOut);

private:
    Net* _pNet;
};

#endif
