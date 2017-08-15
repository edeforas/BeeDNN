#include <iostream>
#include <cmath>
using namespace std;

#include "Net.h"
#include "Activation.h"
#include "DenseLayer.h"
#include "MNISTReader.h"
#include "MatrixUtil.h"

class LossObserver: public TrainObserver
{
public:
    virtual void stepEpoch(const TrainResult & tr)
    {
        cout << "epoch=" << tr.computedEpochs << " loss=" << tr.loss << " maxerror=" << tr.maxError << endl;
    }
};

int main()
{
    Net n;
    LossObserver lo;
    Matrix mRefImages,mRefLabels, mTestImages,mTestLabels;

    cout << "loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabels, mTestImages,mTestLabels))
    {
        cout << "MNIST samples not found, please check the *-ubyte files are in exectuable folder" << endl;
        return -1;
    }

    // normalize input data
    mRefImages=mRefImages/128.-1.;
    mTestImages=mTestImages/128.-1.;

    //transform truth as a probabilty vector (one column by class)
    mRefLabels=index_to_position(mRefLabels,10);
    mTestLabels=index_to_position(mTestLabels,10);

    ActivationManager am;
    DenseLayer l1(784,20,am.get_activation("Tanh"));
    DenseLayer l2(20,10,am.get_activation("Tanh"));
    DenseLayer l3(10,10,am.get_activation("Tanh"));

    n.add(&l1);
    n.add(&l2);
    n.add(&l3);

    TrainOption tOpt;
    tOpt.epochs=10;
    tOpt.earlyAbortMaxError=0.05;
    tOpt.learningRate=0.1;
    tOpt.batchSize=48;
    tOpt.momentum=0.05;
    tOpt.observer=&lo;

    cout << "training..." << endl;

    n.train(mRefImages,mRefLabels,tOpt);

    cout << "end of learning" << endl;

    //todo compute on test db
    //todo compute conf matrix

    return 0;
}
