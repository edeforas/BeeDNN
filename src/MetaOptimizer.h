#ifndef MetaOptimizer_
#define MetaOptimizer_

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

class MetaOptimizer
{
public:
	MetaOptimizer();
	~MetaOptimizer();
	
	void set_net(Net* pNet);
	void set_train(NetTrain* pTrain);


private:
	Net* _pNet;
	NetTrain* _pTrain;
};

#endif