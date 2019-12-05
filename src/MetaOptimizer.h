#ifndef MetaOptimizer_
#define MetaOptimizer_

#include "NetTrain.h"

#include <functional>
using namespace std;

class MetaOptimizer
{
public:
	MetaOptimizer();
	~MetaOptimizer();
	
	void set_train(NetTrain& train);
	void set_nb_thread(int iNbThread); // default: use max available or if iNbThread is zero
	void run();
	void set_better_solution_callback(std::function<void(NetTrain& train)> betterSolutionCallBack);

private:
	static int run_thread(int iThread, MetaOptimizer* self);
	std::function<void(NetTrain& train)> _betterSolutionCallBack;
	NetTrain* _pTrain;
	int _iNbThread;
};

#endif