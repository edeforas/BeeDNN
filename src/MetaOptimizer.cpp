#include "MetaOptimizer.h"

#include "Net.h"

#include <ctime>
#include <thread>

//////////////////////////////////////////////////////////////////////////////
MetaOptimizer::MetaOptimizer()
{
	_fBestAccuracy = -1.;
	_pTrain = nullptr;
	_iNbThread = 0;
	_iNRepeatAll = 1;
	_betterSolutionCallBack = nullptr;
}
//////////////////////////////////////////////////////////////////////////////
MetaOptimizer::~MetaOptimizer()
{ }
//////////////////////////////////////////////////////////////////////////////
void MetaOptimizer::set_train(NetTrain& train)
{
	_pTrain = &train;
}
//////////////////////////////////////////////////////////////////////////////
void MetaOptimizer::set_nb_thread(int iNbThread)
{
	_iNbThread = iNbThread;
}
//////////////////////////////////////////////////////////////////////////////
void MetaOptimizer::set_repeat_all(int iNbRepeatAll)
{
	_iNRepeatAll = iNbRepeatAll;
}
//////////////////////////////////////////////////////////////////////////////
void MetaOptimizer::run()
{
	for (int iRepeat = 0; iRepeat < _iNRepeatAll; iRepeat++)
	{
		_fBestAccuracy = -1.;

		int iNbThread = _iNbThread;
		if (iNbThread == 0) //auto case
			iNbThread = (int)(thread::hardware_concurrency());

		vector<thread> vt(iNbThread);

		for (int i = 0; i < iNbThread; i++)
		{
			srand((unsigned int)time(NULL)); //avoid using the same global rand for every thread
			vt[i] = std::thread(&run_thread, i, this);
		}

		for (int i = 0; i < iNbThread; i++)
			vt[i].join();
	}
}
////////////////////////////////////////////////////////////////
void MetaOptimizer::new_epoch(NetTrain& trainT)
{
	//todo add locks

	if (trainT.get_current_test_accuracy() > _fBestAccuracy)
	{
		_fBestAccuracy = trainT.get_current_test_accuracy();

		_betterSolutionCallBack(trainT);
	}
}
////////////////////////////////////////////////////////////////
int MetaOptimizer::run_thread(int iThread, MetaOptimizer* self)
{
	//hard copy ref net and train
	Net netT;
	NetTrain trainT;
	
	//change rand seed for each threads
	for (int i = 0; i < iThread; i++)
		randomEngine()();

	netT = self->_pTrain->net();
	trainT = *(self->_pTrain);
	
	trainT.set_net(netT);

	// lambda epoch callback:
	trainT.set_epoch_callback([&]()
		{
			//todo call optimizer callback with trainT as arg
		self->new_epoch(trainT);
		}
	);

	trainT.train();

	return 0; 
}
////////////////////////////////////////////////////////////////
void MetaOptimizer::set_better_solution_callback(std::function<void(NetTrain& train)> betterSolutionCallBack)
{
	_betterSolutionCallBack = betterSolutionCallBack;
}
////////////////////////////////////////////////////////////////