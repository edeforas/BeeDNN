#ifndef DNNEngineBeeDnn_
#define DNNEngineBeeDnn_

#include "Net.h"
#include "NetTrain.h"

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
    vector<float> trainLoss;
	vector<float> testLoss;
	vector<float> trainAccuracy;
    vector<float> testAccuracy;
};


class MLEngineBeeDnn
{
public:
    MLEngineBeeDnn();
    virtual ~MLEngineBeeDnn() ;

     void clear() ;
     void init() ;

	Net& net();
    const Net& net() const;

    NetTrain& netTrain();
    const NetTrain& netTrain() const;

     DNNTrainResult learn(const MatrixFloat& mSamples,const MatrixFloat& mTruth);

private:
    Net* _pNet;
    NetTrain* _pTrain;

	//temp
    vector<float> _vfTrainLoss;
	vector<float> _vfTestLoss;
	vector<float> _vfTrainAccuracy;
    vector<float> _vfTestAccuracy;

    int _iComputedEpochs;
};

#endif
