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
        decay=0.9f;
        momentum=0.9f;
        testEveryEpochs=1;
        //observer=0;

    }

    int  epochs;
    int batchSize;

    //optimizer settings
    float learningRate;
    float decay;
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
    vector<double> accuracy;
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
    virtual void add_dense_layer(int inSize,int outSize, bool bWithBias)=0;
    virtual void add_activation_layer(string sActivation) =0;
    virtual void add_dropout_layer(int inSize,float fRatio) =0;

    void set_problem(bool bClassification);
    bool is_classification_problem();
    virtual DNNTrainResult learn(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto);

    virtual void predict(const MatrixFloat& mIn, MatrixFloat& mOut)=0;
    virtual void predict_all(const MatrixFloat & mSamples, MatrixFloat& mResult);

    void classify_all(const MatrixFloat & mSamples, MatrixFloat& mResultLabel);

    virtual float compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth)=0;

    virtual void compute_confusion_matrix(const MatrixFloat & mSamples, const MatrixFloat& mTruth, MatrixFloat& mConfusionMatrix, float& fAccuracy);

protected:	
    virtual void learn_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)=0;
    vector<double> _vdLoss; //temp
    vector<double>_vdAccuracy; //temp

    int _bClassification;
    int _iComputedEpochs;
};

#endif
