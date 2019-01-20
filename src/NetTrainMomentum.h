#ifndef NetTrainMomentum_
#define NetTrainMomentum_

#include <vector>
using namespace std;

#include "Matrix.h"

class Layer;
class Net;
class TrainObserver;

class TrainOption
{
public:
    TrainOption()
    {
        epochs=1000;
        earlyAbortMaxError=0.;
        earlyAbortMeanError=0.;
        batchSize=32;
        learningRate=0.1f;
        momentum=0.1f;
        observer=0;
        initWeight=true;
    }

    int  epochs;
    double earlyAbortMaxError;
    double earlyAbortMeanError;
    int batchSize;
    float learningRate;
    float momentum;
    bool initWeight;
    TrainObserver* observer;
};

class TrainObserver
{
public:
    virtual void stepEpoch(/*const TrainResult & tr*/)=0;
};

class NetTrainMomentum
{
public:
    NetTrainMomentum();
    virtual ~NetTrainMomentum();

    // return the number of epochs,
    int train(Net& net, const MatrixFloat& mSamples, const MatrixFloat& mTruth, const TrainOption& topt);

private:
    void backpropagation(Net& net,const MatrixFloat& mError);
};

#endif
