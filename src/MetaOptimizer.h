#ifndef MetaOptimizer_
#define MetaOptimizer_

#include "NetTrain.h"

class MetaOptimizer
{
public:
	MetaOptimizer();
	~MetaOptimizer();
	
	void set_train(NetTrain& train);
	void set_nb_thread(int iNbThread); // default: use max available or if iNbThread is zero
	void run();
	
private:
	static int run_thread(int iThread, MetaOptimizer* self);
	NetTrain* _pTrain;
	int _iNbThread;
};

#endif