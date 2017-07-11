#ifndef Net_
#define Net_

#include <vector>
using namespace std;

#include "Layer.h"
#include "Matrix.h"

class TrainOption
{
public:
    int  epochs;
    double maxError;
    int batchSize; //32
    double learningRate; //0.1
    double momentum; //0.05
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
