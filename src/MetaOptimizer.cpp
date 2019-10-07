#include "MetaOptimizer.h"

#include <ctime>
#include <thread>

//////////////////////////////////////////////////////////////////////////////
MetaOptimizer::MetaOptimizer()
{
	_pNet = nullptr;
	_pTrain = nullptr;
	_iNbThread = 0;
}
//////////////////////////////////////////////////////////////////////////////
MetaOptimizer::~MetaOptimizer()
{ }
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
void MetaOptimizer::set_nb_thread(int iNbThread)
{
	_iNbThread = iNbThread;
}
//////////////////////////////////////////////////////////////////////////////
void MetaOptimizer::run()
{
	int iNbThread = _iNbThread;
	if(iNbThread==0) //auto case
		iNbThread = (int)(thread::hardware_concurrency());
	
	vector<thread> vt(iNbThread);

	for (int i = 0; i < iNbThread; i++)
	{
		srand((unsigned int)time(NULL)); //avoid using the same global rand for every thread
		vt[i] = std::thread(&run_thread, i,this);
	}

	for (int i = 0; i < iNbThread; i++)
		vt[i].join();
}
////////////////////////////////////////////////////////////////
int MetaOptimizer::run_thread(int iThread, MetaOptimizer* self)
{
	//copy ref net
	Net netT= *(self->_pNet);
	NetTrain trainT;
	trainT= *(self->_pTrain);
	trainT.set_net(netT);
	trainT.train();

	/*
	
	    MatrixFloat mClassRef;
    net.classify_all(mRefImages, mClassRef);
    ConfusionMatrix cmRef;
    ClassificationResult crRef = cmRef.compute(mRefLabels, mClassRef);
    cout << "% accuracy on Ref =" << crRef.accuracy << endl;
	
    MatrixFloat mClassTest;
    net.classify_all(mTestImages, mClassTest);
    ConfusionMatrix cmTest;
    ClassificationResult crTest = cmTest.compute(mTestLabels, mClassTest);
    cout << "% accuracy on Test=" << crTest.accuracy << endl;
	*/





	return 0; //TODO WIP WIP
}
////////////////////////////////////////////////////////////////
