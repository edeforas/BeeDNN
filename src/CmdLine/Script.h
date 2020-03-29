#ifndef _Script_
#define _Script_

#include <string>
#include <vector>
using namespace std;

#include "Net.h"
#include "NetTrain.h"

class Script
{
public:
	Script();

	void run_file(string sFile);
	void run(string sCmd);

private:
	vector<string> cleanup(const string &s);

	Net _net;
	NetTrain _train;

};


#endif