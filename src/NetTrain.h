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
        momentum=0.1f;  //not used for now
		nesterov_momentum = 0.f; //not used for now
        testEveryEpochs=-1;
		sOptimizer = "SGD";
		epochCallBack = nullptr;
    }

    int  epochs;
    int batchSize; //not used for now
    float learningRate;
    float momentum;  //not used for now
	float nesterov_momentum;  //not used for now
	int testEveryEpochs; //set to 1 to test at each epoch, 10 to test only 1/10 of the time, etc, set to -1 for no test
	string sOptimizer; //ex "SGD"
	void(*epochCallBack)();
};

class Layer;
class Net;

class NetTrain
{
public:
    NetTrain();
    virtual ~NetTrain();

    double compute_loss(const Net &net, const MatrixFloat & mSamples, const MatrixFloat& mTruth);

    virtual void train(Net& net,const MatrixFloat& mSamples,const MatrixFloat& mTruthLabel,const TrainOption& topt)=0;
    virtual void fit(Net& net, const MatrixFloat& mSamples, const MatrixFloat& mTruth, const TrainOption& topt)=0;

    vector<double> loss(); //temp

protected:
    vector<double> _vdLoss; //temp
};

#endif
