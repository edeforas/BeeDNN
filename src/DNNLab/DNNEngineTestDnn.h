#ifndef DNNEngineTestDnn_
#define DNNEngineTestDnn_

#include "DNNEngine.h"

class Net;

class DNNEngineTestDnn : public DNNEngine
{
public:
    DNNEngineTestDnn();
    virtual ~DNNEngineTestDnn();
    virtual string to_string() override;

    virtual void clear() override;
    virtual void init() override;
    virtual void add_layer(int inSize,int outSize, string sLayerType) override;

    virtual void train_epochs(const MatrixFloat& mSamples, const MatrixFloat& mTruth, const DNNTrainOption& dto) override;

    virtual void predict(const MatrixFloat& mIn, MatrixFloat& mOut) override;

    virtual double compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth) override;

private:
    Net* _pNet;
};

#endif
