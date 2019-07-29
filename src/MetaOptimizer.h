#ifndef MetaOptimizer_
#define MetaOptimizer_

#include "Net.h"
#include "NetTrain.h"

class MetaOptimizer
{
public:
	MetaOptimizer();
	~MetaOptimizer();
	
	void set_net(Net* pNet);
	void set_train(NetTrain* pTrain);
	void set_nb_thread(int iNbThread); // default: use max available or if iNbThread set to zero
	void run();
	
private:
	static int run_thread(int iThread);
	Net* _pNet;
	NetTrain* _pTrain;
	int _iNbThread;
};

#endif