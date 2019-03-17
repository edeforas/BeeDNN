#ifndef NetTrain_
#define NetTrain_

#include "Matrix.h"

#include <vector>
#include <string>
using namespace std;

class TrainOption
{
public:
    TrainOption()
    {
        epochs=1000;
        batchSize=1; //not used for now
        learningRate=0.01f;
        decay=0.9f;
        momentum=0.9f;
        testEveryEpochs=-1;
        optimizer = "Nesterov";
		epochCallBack = nullptr;
    }

    int  epochs;
    int batchSize; //not used for now
    float learningRate;
    float decay;
    float momentum;
    int testEveryEpochs; //set to 1 to test at each epoch, 10 to test only 1/10 of the time, etc, set to -1 for no test //todo remove
    string optimizer; //ex "SGD" "Momentum" Adam" "Adagrad" "Nesterov" "RMSProp"
    void(*epochCallBack)(); //called after an epoch
};

class Layer;
class Net;

class NetTrain
{
public:
    NetTrain();
    virtual ~NetTrain();

    float compute_loss(const Net &net, const MatrixFloat & mSamples, const MatrixFloat& mTruth);

    void train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruthLabel,const TrainOption& topt);
    void fit(Net& net, const MatrixFloat& mSamples, const MatrixFloat& mTruth, const TrainOption& topt);

    vector<double> loss(); //temp

protected:
    vector<double> _vdLoss; //temp
};

#endif
