#ifndef DNNEngineTestDnn_
#define DNNEngineTestDnn_

#include "DNNEngine.h"

class Net;

class DNNEngineTestDnn : public DNNEngine
{
public:
    DNNEngineTestDnn();
    virtual ~DNNEngineTestDnn();

    virtual void predict(const MatrixFloat mIn, MatrixFloat mOut);

private:
    Net* _pNet;
};

#endif
