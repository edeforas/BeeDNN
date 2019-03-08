#ifndef NetTrainSGD_
#define NetTrainSGD_

#include "Matrix.h"
#include "NetTrain.h"

// train net with Stochastic Gradient Descent method, one weight update by sample

class Net;

class NetTrainSGD : public NetTrain
{
public:
    NetTrainSGD();
    virtual ~NetTrainSGD() override;

    virtual void fit(Net& net, const MatrixFloat& mSamples, const MatrixFloat& mTruth, const TrainOption& topt) override ;
};

#endif
