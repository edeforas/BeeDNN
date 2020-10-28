#include "MetaOptimizer.h"

#include "Net.h"

#include "LayerFactory.h"

#include <iostream>
#include <ctime>
#include <thread>

//////////////////////////////////////////////////////////////////////////////
MetaOptimizer::MetaOptimizer():
	_betterSolutionCallBack(nullptr)
{
	_fBestAccuracy = -1.;
	_pTrain = nullptr;
	_iNbThread = 0;
	_iNRepeatAll = 1;
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
	int iNbThread = _iNbThread;
	if (iNbThread == 0) //auto case
		iNbThread = (int)(thread::hardware_concurrency());

	for (int iRepeat = 0; iRepeat < _iNRepeatAll; iRepeat++)
	{
		cout << "Restart " << iRepeat << "/" << _iNRepeatAll << endl;

		_fBestAccuracy = -1.;

		vector<thread> vt(iNbThread);

		for (int i = 0; i < iNbThread; i++)
		{
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

	if (trainT.get_current_validation_accuracy() > _fBestAccuracy)
	{
		_fBestAccuracy = trainT.get_current_validation_accuracy();

		_betterSolutionCallBack(trainT);
	}
}
////////////////////////////////////////////////////////////////
int MetaOptimizer::run_thread(int iThread, MetaOptimizer* self)
{
	//change rand seed for each threads
	for (int i = 0; i < iThread; i++)
		randomEngine()();

	//hard copy ref net and train
	Net netT;
	NetTrain trainT;
	netT = self->_pTrain->net();
	trainT = *(self->_pTrain);
	trainT.set_net(netT);

	self->apply_variations(netT);

	// lambda epoch callback:
	trainT.set_epoch_callback([&]()
		{
			//todo call optimizer callback with trainT as arg
		self->new_epoch(trainT);
		}
	);

	trainT.fit();

	return 0; 
}
////////////////////////////////////////////////////////////////
void MetaOptimizer::set_better_solution_callback(std::function<void(NetTrain& train)> betterSolutionCallBack)
{
	_betterSolutionCallBack = betterSolutionCallBack;
}
////////////////////////////////////////////////////////////////
void MetaOptimizer::add_variation(Index iLayer,const string & sType, float fArg1, float fArg2, float fArg3, float fArg4, float fArg5)
{
	MetaOptimizerVariation v;
	v.iLayer = iLayer;
	v.sType = sType;
	v.fArg1 = fArg1;
	v.fArg2 = fArg2;
	v.fArg3 = fArg3;
	v.fArg4 = fArg4;
	v.fArg5 = fArg5;

	_variations.push_back(v);
}
////////////////////////////////////////////////////////////////
void MetaOptimizer::apply_variations(Net& net)
{
	// set optional variation
	for (size_t iL = 0; iL < net.size(); iL++)
	{
		//collect all variations for a layer, not optimized, but ok
		vector<MetaOptimizerVariation> vl;
		for (size_t iv = 0; iv < _variations.size(); iv++)
		{
			if (_variations[iv].iLayer == (Index)iL)
				vl.push_back(_variations[iv]);
		}

		auto iVariation = randomEngine()() % (vl.size()+1);

		if (iVariation > 0) //original variation accepted
		{
			const MetaOptimizerVariation & v = vl[iVariation-1];

			net.replace(iL, LayerFactory::create(v.sType,v.fArg1, v.fArg2, v.fArg3, v.fArg4, v.fArg5));
		}
	}
}
////////////////////////////////////////////////////////////////


