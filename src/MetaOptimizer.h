#ifndef MetaOptimizer_
#define MetaOptimizer_

#include "NetTrain.h"

#include <functional>
#include <vector>
using namespace std;

class MetaOptimizerVariation
{
public:
	Index iLayer;
	string sType;
	float fArg1;
	float fArg2;
	float fArg3;
	float fArg4;
	float fArg5;

	MetaOptimizerVariation()
	{
		iLayer = 0;
		sType = "";
		fArg1=0.f;
		fArg2 = 0.f;
		fArg3 = 0.f;
		fArg4 = 0.f;
		fArg5 = 0.f;
	}
};


class MetaOptimizer
{
public:
	MetaOptimizer();
	~MetaOptimizer();
	
	void set_train(NetTrain& train);
	void set_nb_thread(int iNbThread); // default: use max available or if iNbThread is zero
	
	void add_variation(Index iLayer,const string&  sType, float fArg1 = 0.f, float fArg2 = 0.f, float fArg3 = 0.f, float fArg4 = 0.f, float fArg5 = 0.f);

	void set_repeat_all(int iNbRepeatAll);
	
	void run();
	void set_better_solution_callback(std::function<void(NetTrain& train)> betterSolutionCallBack);

private:
	void new_epoch(NetTrain& trainT);
	static int run_thread(int iThread, MetaOptimizer* self);
	std::function<void(NetTrain& train)> _betterSolutionCallBack;
	
	NetTrain* _pTrain;
	int _iNbThread;
	int _iNRepeatAll;
	float _fBestAccuracy;

	void apply_variations(Net& net);
	vector< MetaOptimizerVariation> _variations;
};

#endif