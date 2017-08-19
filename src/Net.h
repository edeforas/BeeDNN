#ifndef Net_
#define Net_

#include <vector>
using namespace std;

class Layer;
#include "Matrix.h"

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
        learningRate=0.1;
        momentum=0.1;
        subSamplingRatio=1;
        observer=0;
    }

    int  epochs;
    double earlyAbortMaxError;
    double earlyAbortMeanError;
    int batchSize;
    double learningRate;
    double momentum;
    int subSamplingRatio; //1 to keep everything in training ; 2 , to keep half (shuffled) and soon on
    TrainObserver* observer;
};

class TrainResult
{
public:
    double loss;
    double maxError;
    int computedEpochs;
};

class TrainObserver
{
public:
    virtual void stepEpoch(const TrainResult & tr)=0;
};

class Net
{
public:
    Net();
    virtual ~Net();

    void add(Layer *l);

    TrainResult train(const Matrix& mSamples, const Matrix& mTruth, const TrainOption& topt);
    void forward(const Matrix& mIn,Matrix& mOut) const;

    void classify(const Matrix& mIn,Matrix& mClass) const; // todo move in classification problem

private:
    void backpropagation(const Matrix& mError);
    vector<Layer*> _layers;
};

#endif
