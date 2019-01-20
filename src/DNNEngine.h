#ifndef DNNEngine_
#define DNNEngine_

#include <string>
using namespace std;

#include "Matrix.h"

enum eLayerType
{
    FullyConnected=1
};

class DNNTrainObserver;

class DNNTrainOption
{
public:
    DNNTrainOption()
    {
        epochs=1000;
        batchSize=32;
        learningRate=0.1f;
        momentum=0.1f;
        initWeight=true;
        //observer=0;
    }

    int  epochs;
    double earlyAbortMaxError;
    double earlyAbortMeanError;
    int batchSize;

    //momentum settings
    float learningRate;
    float momentum;
    bool initWeight;

    DNNTrainObserver* observer;
};

class DNNTrainResult
{
public:
    double loss;
    double maxError;
    int computedEpochs;
    double epochDuration; //in second
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

    virtual void clear()=0;
    virtual void init();
    virtual void add_layer_and_activation(int inSize,int outSize, eLayerType layer, string sActivation)=0;

    virtual DNNTrainResult train(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto);
    virtual void train_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)=0;

    virtual void predict(const MatrixFloat& mIn, MatrixFloat& mOut)=0;

private:
    int _iComputedEpochs;
};

#endif
