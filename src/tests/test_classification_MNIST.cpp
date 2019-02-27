#include <iostream>
using namespace std;

#include "Net.h"
#include "NetTrainSGD.h"
#include "MNISTReader.h"
#include "MatrixUtil.h"
#include "ConfusionMatrix.h"

Net net;
MatrixFloat mRefImages, mRefLabelsIndex, mTestImages, mTestLabelsIndex;
int iEpoch;

//////////////////////////////////////////////////////////////////////////////
void disp(const MatrixFloat& m)
{
    for(unsigned int r=0;r<m.rows();r++)
    {
        for(unsigned int c=0;c<m.cols();c++)
            cout << m(r,c) << " ";
        cout << endl;
    }
}
//////////////////////////////////////////////////////////////////////////////
void epoch_callback()
{
	iEpoch++;
	cout << " epoch:" << iEpoch << endl;

	// perfs on learning dDB
	{
		cout << " result on full learning DB:" << endl;
		MatrixFloat mClass;
		net.classify_all(mRefImages, mClass);

		ConfusionMatrix cm;
		ClassificationResult cr = cm.compute(mRefLabelsIndex, mClass, 10);

		cout << "% of good detection=" << cr.goodclassificationPercent << endl;

		cout << "ConfusionMatrix=" << endl;
		disp(cr.mConfMat);
		cout << endl;
	}

	// perfs on test dDB
	{
		cout << " result on full test DB:" << endl;
		MatrixFloat mClass;
		net.classify_all(mTestImages, mClass);

		ConfusionMatrix cm;
		ClassificationResult cr = cm.compute(mTestLabelsIndex, mClass, 10);

		cout << "% of gooddetection=" << cr.goodclassificationPercent << endl;

		cout << "ConfusionMatrix=" << endl;
		disp(cr.mConfMat);
		cout << endl;
	}
}
//////////////////////////////////////////////////////////////////////////////
int main()
{
	iEpoch = 0;

    //TODO update sample

    cout << "loading MNIST database..." << endl;
    MNISTReader mr;
    if(!mr.read_from_folder(".",mRefImages,mRefLabelsIndex, mTestImages,mTestLabelsIndex))
    {
        cout << "MNIST samples not found, please check the *-ubyte files are in executable folder" << endl;
        return -1;
    }

    //normalize data
    mTestImages/=256.f;
    mRefImages/=256.f;

    //simple net: 97% classif 76% test after a long time
    net.add_layer("DenseAndBias",784,512);
	net.add_dropout_layer(512,0.5f);
    net.add_layer("Relu",512,512);
    net.add_layer("DenseAndBias",512,10);
	net.add_dropout_layer(10,0.5f);
    net.add_layer("Sigmoid",10,10);

    TrainOption tOpt;
    tOpt.epochs=100;
	tOpt.epochCallBack = epoch_callback;

    cout << "training..." << endl;

    NetTrainSGD netTrain;
    netTrain.train(net,mRefImages,mRefLabelsIndex,tOpt); //todo rewrite

    cout << "end of test." << endl;
    return 0;
}
