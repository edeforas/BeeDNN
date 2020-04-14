#ifndef NetScript_
#define NetScript_

#include <string>
#include <vector>
using namespace std;

#include "Net.h"
#include "NetTrain.h"
#include "DataSource.h"

class NetScript
{
public:
	NetScript();

	void run_file(string sFile);
	void run(string sCmd);

private:
	vector<string> cleanup(const string &s);

	Net _net;
	NetTrain _train;
	DataSource _data;
};


#endif