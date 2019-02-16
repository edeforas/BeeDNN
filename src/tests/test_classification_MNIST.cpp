#include <iostream>
using namespace std;

#include "Net.h"
#include "NetTrainLearningRate.h"
#include "MNISTReader.h"
#include "MatrixUtil.h"
#include "ConfusionMatrix.h"

Net net;
MatrixFloat mRefImages, mRefLabels, mRefLabelsIndex, mTestImages, mTestLabels, mTestLabelsIndex;

void disp(const MatrixFloat& m)
{
    for(unsigned int r=0;r<m.rows();r++)
    {
        for(unsigned int c=0;c<m.cols();c++)
            cout << m(r,c) << " ";
        cout << endl;
    }
}
/*
class LossObserver: public TrainObserver
{
public:
    virtual void stepEpoch()
    {
    //    cout << "epoch=" << tr.computedEpochs << " duration=" << tr.epochDuration << "s loss=" << tr.loss << " maxerror=" << tr.maxError << endl;
		
        MatrixFloat mClass;
        net.classify(mTestImages,mClass);

        ConfusionMatrix cm;
        ClassificationResult cr=cm.compute(mTestLabelsIndex,mClass,10);

        cout << "% of good detection=" << cr.goodclassificationPercent << endl;

        cout << "ConfusionMatrix=" << endl;
        disp(cr.mConfMat);
        cout << endl;
    }
};
*/
int main()
{
    //LossObserver lo;

    //TODO update sample

    cout << "loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabelsIndex, mTestImages,mTestLabelsIndex))
    {
        cout << "MNIST samples not found, please check the *-ubyte files are in executable folder" << endl;
        return -1;
    }

    //transform truth as a probability vector (one column by class)
    mRefLabels=index_to_position(mRefLabelsIndex,10);
    mTestLabels=index_to_position(mTestLabelsIndex,10);

    //normalize data
    mTestImages/=256.f-0.5f;
    mRefImages/=256.f-0.5f;

    //simple net, expect only 33% good classification with it ...
    net.add_layer("DenseAndBias",784,30);
    net.add_layer("Tanh",30,30);
    net.add_layer("DenseAndBias",30,10);
    net.add_layer("Tanh",10,10);

    TrainOption tOpt;
    tOpt.epochs=1000;
    tOpt.learningRate=0.1f;
    tOpt.batchSize=128;
    //tOpt.observer=&lo;

    cout << "training..." << endl;

    NetTrainLearningRate netTrain;
    netTrain.train(net,mRefImages,mRefLabels,tOpt);

    cout << "end of training." << endl;
    cout << "computing perfs ..."<< endl;

    // perfs on learning dDB
    {
        cout << " result on full learning DB:" << endl;
        MatrixFloat mClass;
        net.classify(mRefImages,mClass);

        ConfusionMatrix cm;
        ClassificationResult cr=cm.compute(mRefLabelsIndex,mClass,10);

        cout << "% of good detection=" << cr.goodclassificationPercent << endl;

        cout << "ConfusionMatrix=" << endl;
        disp(cr.mConfMat);
        cout << endl;
    }

    // perfs on test dDB
    {
        cout << " result on full test DB:" << endl;
        MatrixFloat mClass;
        net.classify(mTestImages,mClass);

        ConfusionMatrix cm;
        ClassificationResult cr=cm.compute(mTestLabelsIndex,mClass,10);

        cout << "% of gooddetection=" << cr.goodclassificationPercent << endl;

        cout << "ConfusionMatrix=" << endl;
        disp(cr.mConfMat);
        cout << endl;
    }

    cout << "end of test." << endl;
    return 0;
}
