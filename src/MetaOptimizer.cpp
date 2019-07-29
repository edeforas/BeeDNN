#include "MetaOptimizer.h"

//////////////////////////////////////////////////////////////////////////////
MetaOptimizer::MetaOptimizer()
{
	_pNet = nullptr;
	_pTrain = nullptr;
}
//////////////////////////////////////////////////////////////////////////////
MetaOptimizer::~MetaOptimizer()
{

}
//////////////////////////////////////////////////////////////////////////////
void MetaOptimizer::set_net(Net* pNet)
{
	_pNet = pNet;
}
//////////////////////////////////////////////////////////////////////////////
void MetaOptimizer::set_train(NetTrain* pTrain)
{
	_pTrain = pTrain;
}
//////////////////////////////////////////////////////////////////////////////


#if 0

#include <iostream>
#include <fstream>
#include <ctime>
#include <thread>
#include <string>
#include <vector>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "NetUtil.h"
#include "Matrix.h"

#include "ConfusionMatrix.h"

MatrixFloat mTrainData, mTrainTruthLabel;

//////////////////////////////////////////////////////////////////////////////
void save_solution(Net& net, NetTrain& train, string sFile)
{
	string s;
	NetUtil::write(train, s);
	NetUtil::write(net, s);

	ofstream out(sFile, ios::binary);
	out << s;

	cout << sFile << endl;
	cout << "Matconf_test: " << endl ;
	cout << "---------------------------------------------------------" << endl;
}
//////////////////////////////////////////////////////////////////////////////
void main_thread(int iThread)
{
	srand(iThread+(unsigned int)time(NULL));

	vector<string> vsOptimizers;
    vsOptimizers.push_back("Adam");
    vsOptimizers.push_back("Nadam");
    vsOptimizers.push_back("Adamax");

	vector<string> vsActivations;
	vsActivations.push_back("Relu");
	vsActivations.push_back("Relu6");
	vsActivations.push_back("LeakyRelu256");
	vsActivations.push_back("LeakyTwiceRelu6");
	vsActivations.push_back("SQNL");
	vsActivations.push_back("Parablu");

    while(true)
    {
		int iOptimizer = rand() % vsOptimizers.size();
		int iActivation1 = rand() % vsActivations.size();
		int iActivation2 = rand() % vsActivations.size();
		int iActivationOut = rand() % vsActivations.size();

		int iReboost = (rand() % 11)+20; //from 20 up to 30 reboost
		int iNbNeurons = (rand() % 10) + 30; //20 to 30 neurons
		int iEpochs= (rand() % 50) + 80; //from 80 to 130 epochs
		string sActivation1 = vsActivations[iActivation1];
		string sActivation2 = vsActivations[iActivation2];
		string sActivationOut = vsActivations[iActivationOut];
		string sOptimizer = vsOptimizers[iOptimizer];

		Net net;
		net.add_dense_layer((int)mTrainData.cols(), iNbNeurons);
		net.add_activation_layer(sActivation1);
		net.add_dense_layer(iNbNeurons, iNbNeurons);
		net.add_activation_layer(sActivation2);
		net.add_dense_layer(iNbNeurons, 1);
		net.add_activation_layer(sActivationOut);

        //train
        NetTrain netTrain;
		netTrain.set_epochs(iEpochs);
		netTrain.set_optimizer(sOptimizer);
		netTrain.set_keepbest(true);

		//train!
		netTrain.set_reboost_every_epochs(iReboost);
		netTrain.train(net, mTrainData, mTrainTruthLabel);
			
        //compute results
        MatrixFloat mTrainClass;
        net.classify_all(mTrainData,mTrainClass);
        ConfusionMatrix cm;
        ClassificationResult cr=cm.compute(mTrainTruthLabel, mTrainClass);

		if (cr.accuracy > minAccuracy)
		{
			stringstream ss;
			ss << "Acc" << cr.accuracy << "_" << sOptimizer << "_" << sActivation1 << "_NN" << iNbNeurons << "_" << sActivation2 << "_" << sActivationOut << "_Epochs" << iEpochs << "_Reboost" << iReboost << ".dnnlab";
			cout << ss.str() << endl;

			save_solution(net, netTrain, ss.str());

			cout << "Matconf_test: " << endl << toString(cr.mConfMat) << endl;
			cout << "---------------------------------------------------------" << endl;
		}
		else
			cout << cr.accuracy << "%" << endl;
    }

	cout << "end of thread" << endl;
}

//////////////////////////////////////////////////////////////////////////////
int main()
{

	//load data

	mTrainData = fromFile("data.csv");
	mTrainTruthLabel = fromFile("truth.csv");
	
	//mTrainData /= 256.f;

	if (mTrainData.rows() != mTrainTruthLabel.rows())
	{
		cout << "Error, files do not have the same number of samples" << endl;
		return -1;
	}


	int iNbThread = (int)(thread::hardware_concurrency());
	if (iNbThread < 0)
		iNbThread = 1;

	cout << "Starting " << iNbThread << " threads" << endl;
	vector<thread> vt(iNbThread);

	for (int i = 0; i < iNbThread; i++)
		vt[i] = std::thread(main_thread,i);

	for (int i = 0; i < iNbThread; i++)
		vt[i].join();

}
////////////////////////////////////////////////////////////////

#endif