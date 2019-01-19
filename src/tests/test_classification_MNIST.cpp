#include <iostream>
using namespace std;

#include "Net.h"
#include "Activation.h"
#include "ActivationLayer.h"
#include "MNISTReader.h"
#include "MatrixUtil.h"
#include "ConfusionMatrix.h"

Net n;
MatrixFloat mRefImages, mRefLabels, mRefLabelsIndex, mTestImages, mTestLabels, mTestLabelsIndex;

void disp(const MatrixFloat& m)
{
    for(unsigned int r=0;r<m.rows();r++)
    {
        for(unsigned int c=0;c<m.columns();c++)
            cout << m(r,c) << " ";
        cout << endl;
    }
}
/*
class LossObserver: public TrainObserver
{
public:
    virtual void stepEpoch(const TrainResult & tr)
    {
        cout << "epoch=" << tr.computedEpochs << " duration=" << tr.epochDuration << "s loss=" << tr.loss << " maxerror=" << tr.maxError << endl;
		
        MatrixFloat mClass;
        n.classify(mTestImages,mClass);

        ConfusionMatrix cm;
        ClassificationResult cr=cm.compute(mTestLabelsIndex,mClass,10);

        cout << "% of good detection=" << cr.goodclassificationPercent << endl;

        cout << "ConfusionMatrixFloat=" << endl;
        disp(cr.mConfMat);
        cout << endl;
    }
};
*/
int main()
{
    //LossObserver lo;

    cout << "loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabelsIndex, mTestImages,mTestLabelsIndex))
    {
        cout << "MNIST samples not found, please check the *-ubyte files are in executable folder" << endl;
        return -1;
    }

    // normalize input data
    mTestImages/=256.f;
    mRefImages/=256.f;

    //transform truth as a probabilty vector (one column by class)
    mRefLabels=index_to_position(mRefLabelsIndex,10);
    mTestLabels=index_to_position(mTestLabelsIndex,10);

    ActivationLayer l1(784,200,"Tanh");
    ActivationLayer l2(200,50,"Tanh");
    ActivationLayer l3(50,10,"Tanh");

    n.add(&l1);
    n.add(&l2);
    n.add(&l3);

    TrainOption tOpt;
    tOpt.epochs=100;
    tOpt.earlyAbortMaxError=0.05;
    tOpt.learningRate=0.2f;
    tOpt.batchSize=128;
    tOpt.momentum=0.1f;
    //tOpt.observer=&lo;

    cout << "training..." << endl;

    n.train(mRefImages,mRefLabels,tOpt);

    cout << "end of training." << endl;
    cout << "computing perfs ..."<< endl;

    // perfs on learning dDB
    {
        cout << " result on full learning DB:" << endl;
        MatrixFloat mClass;
        n.classify(mRefImages,mClass);

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
        n.classify(mTestImages,mClass);

        ConfusionMatrix cm;
        ClassificationResult cr=cm.compute(mTestLabelsIndex,mClass,10);

        cout << "% of gooddetection=" << cr.goodclassificationPercent << endl;

        cout << "ConfusionMatrixFloat=" << endl;
        disp(cr.mConfMat);
        cout << endl;
    }

    cout << "end of test." << endl;
    return 0;
}
