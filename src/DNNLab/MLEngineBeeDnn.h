#ifndef DNNEngineBeeDnn_
#define DNNEngineBeeDnn_

#include "Net.h"

class Net;
class NetTrain;







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






class MLEngineBeeDnn
{
public:
    MLEngineBeeDnn();
    virtual ~MLEngineBeeDnn() ;

    virtual void write(string& s) ;
    virtual void read(const string&) ;

    virtual void clear() ;
    virtual void init() ;

    virtual void learn_epochs(const MatrixFloat& mSamples, const MatrixFloat& mTruth, const DNNTrainOption& dto) ;

    virtual void predict(const MatrixFloat& mIn, MatrixFloat& mOut) ;

    Net& net();
    const Net& net() const;


    void set_problem(bool bClassification); //classification or regression
    bool is_classification_problem();

    virtual DNNTrainResult learn(const MatrixFloat& mSamples,const MatrixFloat& mTruth,const DNNTrainOption& dto);

    virtual void predict_all(const MatrixFloat & mSamples, MatrixFloat& mResult);
    void classify_all(const MatrixFloat & mSamples, MatrixFloat& mResultLabel);
    virtual float compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth);
    virtual void compute_confusion_matrix(const MatrixFloat & mSamples, const MatrixFloat& mTruth, MatrixFloat& mConfusionMatrix, float& fAccuracy);







private:
    Net* _pNet;
    NetTrain* _pTrain;

    vector<double> _vdLoss; //temp
    vector<double> _vdAccuracy; //temp

    int _bClassification;
    int _iComputedEpochs;
};

#endif
