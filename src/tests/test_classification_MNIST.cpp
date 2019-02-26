#include <iostream>
using namespace std;

#include "Net.h"
#include "NetTrainSGD.h"
#include "MNISTReader.h"
#include "MatrixUtil.h"
#include "ConfusionMatrix.h"

Net net;
MatrixFloat mRefImages, mRefLabelsIndex, mTestImages, mTestLabelsIndex;

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

    //normalize data
    mTestImages/=256.f-0.5f;
    mRefImages/=256.f-0.5f;

    //simple net: 83% classif 70% test
    net.add_layer("DenseAndBias",784,100);
    net.add_layer("Relu",100,100);
    net.add_layer("DenseAndBias",100,10);
    net.add_layer("Sigmoid",10,10);

    TrainOption tOpt;
    tOpt.epochs=30;
    tOpt.testEveryEpochs=1000;

    cout << "training..." << endl;

    NetTrainSGD netTrain;
    netTrain.train(net,mRefImages,mRefLabelsIndex,tOpt); //todo rewrite

    cout << "end of training." << endl;
    cout << "computing perfs ..."<< endl;

    // perfs on learning dDB
    {
        cout << " result on full learning DB:" << endl;
        MatrixFloat mClass;
        net.classify_all(mRefImages,mClass);

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
        net.classify_all(mTestImages,mClass);

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
