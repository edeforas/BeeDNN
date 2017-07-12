#ifndef Net_
#define Net_

#include <vector>
using namespace std;

#include "Layer.h"
#include "Matrix.h"

class TrainOption
{
public:
    TrainOption()
    {
        epochs=1000;
        maxError=0;
        batchSize=32;
        learningRate=0.1;
        momentum=0.05;
    }

    int  epochs;
    double maxError;
    int batchSize;
    double learningRate;
    double momentum;
};

class TrainResult
{
public:
    double loss;
    double maxError;
    int maxEpoch;
};

class Net
{
public:
    Net();
    virtual ~Net();

    void add(Layer *l);

    TrainResult train(const Matrix& mSamples,const Matrix& mTruth,const TrainOption& topt,bool bInit=true);
    void forward(const Matrix& mIn,Matrix& mOut) const;

private:
    void forward_feed(const Matrix& mIn,Matrix& mOut);
    void backpropagation(const Matrix& mError,double dlearningRate);
    vector<Layer*> _layers;
};

#endif
