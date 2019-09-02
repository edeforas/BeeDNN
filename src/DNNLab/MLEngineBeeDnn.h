#ifndef DNNEngineBeeDnn_
#define DNNEngineBeeDnn_

#include "Net.h"
#include "NetTrain.h"
#include "Layer.h"

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
    vector<float> loss;
    vector<float> trainAccuracy;
    vector<float> testAccuracy;
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

    virtual void learn_epochs(const MatrixFloat& mSamples, const MatrixFloat& mTruth) ;

    virtual void predict(const MatrixFloat& mIn, MatrixFloat& mOut) ;

    Net& net();
    const Net& net() const;

    NetTrain& netTrain();
    const NetTrain& netTrain() const;

    void set_classification_mode(bool bClassification); //classification or regression
    bool is_classification_mode();

    virtual DNNTrainResult learn(const MatrixFloat& mSamples,const MatrixFloat& mTruth);

    virtual void predict_all(const MatrixFloat & mSamples, MatrixFloat& mResult);
    void classify_all(const MatrixFloat & mSamples, MatrixFloat& mResultLabel);
    virtual float compute_loss(const MatrixFloat & mSamples, const MatrixFloat& mTruth);
    virtual void compute_confusion_matrix(const MatrixFloat & mSamples, const MatrixFloat& mTruth, MatrixFloat& mConfusionMatrix, float& fAccuracy);

private:
    Net* _pNet;
    NetTrain* _pTrain;

    vector<float> _vfLoss; //temp
    vector<float> _vfTrainAccuracy; //temp
    vector<float> _vfTestAccuracy; //temp

    int _iComputedEpochs;
};

#endif
