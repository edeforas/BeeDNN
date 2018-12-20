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
        learningRate=0.1f;
        momentum=0.1f;
        subSamplingRatio=1;
        observer=0;
    }

    int  epochs;
    double earlyAbortMaxError;
    double earlyAbortMeanError;
    int batchSize;
    float learningRate;
    float momentum;
    int subSamplingRatio; //1 to keep everything in training ; 2 , to keep half (shuffled) and soon on
    TrainObserver* observer;
};

class TrainResult
{
public:
    double loss;
    double maxError;
    int computedEpochs;
    double epochDuration; //in second
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

	void clear();
    void add(Layer *l);
	
    TrainResult train(const MatrixFloat& mSamples, const MatrixFloat& mTruth, const TrainOption& topt);
    void forward(const MatrixFloat& mIn,MatrixFloat& mOut) const;

    void classify(const MatrixFloat& mIn,MatrixFloat& mClass) const; // todo move in classification problem

private:
    void backpropagation(const MatrixFloat& mError);
    vector<Layer*> _layers;
};

#endif
