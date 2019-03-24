#ifndef DNNEngineBeeDnn_
#define DNNEngineBeeDnn_

#include "MLEngine.h"

class Net;

class MLEngineBeeDnn : public MLEngine
{
public:
    MLEngineBeeDnn();
    virtual ~MLEngineBeeDnn();
    virtual string to_string() override;

    virtual void clear() override;
    virtual void init() override;
    virtual void add_dense_layer(int inSize,int outSize, bool bWithBias) override;
    virtual void add_activation_layer(string sActivation) override;
    virtual void add_dropout_layer(int inSize,float fRatio) override;

    virtual void learn_epochs(const MatrixFloat& mSamples, const MatrixFloat& mTruth, const DNNTrainOption& dto) override;

    virtual void predict(const MatrixFloat& mIn, MatrixFloat& mOut) override;

 //   virtual float compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth) override;

private:
    Net* _pNet;
};

#endif
