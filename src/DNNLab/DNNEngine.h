#ifndef DNNEngine_
#define DNNEngine_

#include <string>
#include <vector>
using namespace std;

// DNNEngine is an abstraction class for DNN frameworks

#include "Matrix.h"

//class DNNTrainObserver;

class DNNTrainOption
{
public:
    DNNTrainOption():
	    optimizer("simpleSGD"),
        lossFunction("mse")
    {
        epochs=1000;
        batchSize=32;
        learningRate=0.1f;
        momentum=0.9f;
        testEveryEpochs=1;
        //observer=0;

    }

    int  epochs;
    int batchSize;

    //optimizer settings
    float learningRate;
    float momentum;
    string optimizer;
    string lossFunction;
    int testEveryEpochs; //set to 1 to test at each epoch, 10 to test only 1/10 of the time, etc
//    DNNTrainObserver* observer;
};

class DNNTrainResult
{
public:
    DNNTrainResult()
    {
        finalLoss=0;
        computedEpochs=0;
        epochDuration=-1;
    }

    double finalLoss;
    int computedEpochs;
    double epochDuration; //in second
    vector<double> loss;
};

class DNNTrainObserver
{
public:
    virtual void stepEpoch(const DNNTrainResult & tr)=0;
};

class DNNEngine
{
public:
    DNNEngine();
    virtual ~DNNEngine();
    virtual string to_string()=0;

    virtual void clear()=0; //remove all layers
    virtual void init(); // init weights
    virtual void add_layer(int inSize,int outSize, string sLayerType)=0;

    virtual DNNTrainResult train(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto);

    virtual void predict(const MatrixFloat& mIn, MatrixFloat& mOut)=0;

    virtual double compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth)=0;

protected:	
    virtual void train_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)=0;
    vector<double> _vdLoss; //temp

private:
    int _iComputedEpochs;

};

#endif
