#ifndef DNNEngineTestDnn_
#define DNNEngineTestDnn_

#include "DNNEngine.h"

class Net;

class DNNEngineTestDnn : public DNNEngine
{
public:
    DNNEngineTestDnn();
    virtual ~DNNEngineTestDnn();
    virtual string to_string();

    virtual void clear();
    virtual void init();
    virtual void add_layer(int inSize,int outSize, string sLayerType);

    virtual void train_epochs(const MatrixFloat& mSamples, const MatrixFloat& mTruth, const DNNTrainOption& dto);

    virtual void predict(const MatrixFloat& mIn, MatrixFloat& mOut);

private:
    Net* _pNet;
};

#endif
