#ifndef NetTrainLearningRate_
#define NetTrainLearningRate_

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
        batchSize=32;
        learningRate=0.1f;
    //    momentum=0.1f;
        observer=nullptr;
    }

    int  epochs;
    int batchSize;
    float learningRate;
 //   float momentum;
    TrainObserver* observer;
};

class TrainObserver
{
public:
    virtual void stepEpoch(/*const TrainResult & tr*/)=0;
};

class NetTrainLearningRate
{
public:
    NetTrainLearningRate();
    virtual ~NetTrainLearningRate();

    void train(Net& net, const MatrixFloat& mSamples, const MatrixFloat& mTruth, const TrainOption& topt);
};

#endif
