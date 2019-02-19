#ifndef DNNEngine_
#define DNNEngine_

#include <string>
#include <vector>
using namespace std;

#include "Matrix.h"

//class DNNTrainObserver;

class DNNTrainOption
{
public:
    DNNTrainOption()
    {
        epochs=1000;
        batchSize=32;
        learningRate=0.1f;
        momentum=0.9f;
    //    initWeight=true;
        //observer=0;
        optimizer="simpleSGD";
        lossFunction="mse";
    }

    int  epochs;
 //   double earlyAbortMaxError;
  //  double earlyAbortMeanError;
    int batchSize;

    //momentum settings
    float learningRate;
    float momentum;
    string optimizer;
    string lossFunction;
   // bool initWeight;

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
 //   double maxError;
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

    virtual double compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth);

protected:	
    virtual void train_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)=0;
    vector<double> _vdLoss;

private:
    int _iComputedEpochs;

};

#endif
