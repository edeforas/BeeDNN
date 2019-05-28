#ifndef DNNEngine_
#define DNNEngine_

#include <string>
#include <vector>
using namespace std;

// MLEngine is an abstraction class for all classifiers, DNN specialized for now

#include "Matrix.h"

class DNNTrainOption
{
public:
    DNNTrainOption():
        optimizer("SGD"),
        lossFunction("mse")
    {
        epochs=100;
        batchSize=1;
        keepBest=false;

        learningRate=0.01f;
        decay=0.9f;
        momentum=0.9f;
        reboostEveryEpoch=-1;
    }

    int  epochs;
    int batchSize;
    bool keepBest;

    //optimizer settings
    float learningRate;
    float decay;
    float momentum;
    string optimizer;
    string lossFunction;
    int reboostEveryEpoch;
    int testEveryEpochs; //set to 1 to test at each epoch, 10 to test only 1/10 of the time, etc
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

class MLEngine
{
public:
    MLEngine();
    virtual ~MLEngine();
    virtual void write(string& sFileName)=0;
    virtual void read(const string& sFileName)=0;

    virtual void clear()=0; //remove all layers
    virtual void init(); // init weights

    virtual void add_dense_layer(int inSize,int outSize, bool bWithBias)=0;
    virtual void add_activation_layer(string sActivation) =0;
    virtual void add_dropout_layer(int inSize,float fRatio) =0;
    virtual void add_gaussian_noise_layer(int inSize,float fStd) =0;
    virtual void add_globalgain_layer(int inSize,float fGain) =0;
    virtual void add_poolaveraging1D_layer(int inSize,int iWindowSize) =0;

    void set_problem(bool bClassification); //classification or regression
    bool is_classification_problem();

    virtual DNNTrainResult learn(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto);

    virtual void predict(const MatrixFloat& mIn, MatrixFloat& mOut)=0;

    virtual void predict_all(const MatrixFloat & mSamples, MatrixFloat& mResult);
    void classify_all(const MatrixFloat & mSamples, MatrixFloat& mResultLabel);
    virtual float compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth);
    virtual void compute_confusion_matrix(const MatrixFloat & mSamples, const MatrixFloat& mTruth, MatrixFloat& mConfusionMatrix, float& fAccuracy);

protected:	
    virtual void learn_epochs(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto)=0;
    vector<double> _vdLoss; //temp
    vector<double> _vdAccuracy; //temp

    int _bClassification;
    int _iComputedEpochs;
};

#endif
