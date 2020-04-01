#include "Script.h"

#include <fstream>
#include <iostream>
#include <stdlib.h> // for strtof

#include "NetUtil.h"
#include "LayerFactory.h"

///////////////////////////////////////////////////////////////
Script::Script()
{
	_train.set_net(_net);
}
///////////////////////////////////////////////////////////////
void Script::run_file(string sFile)
{
	// 2 passes to free the file asap
	vector<string> vs;
	{
		ifstream f(sFile);
		while (!f.eof())
		{
			string s;
			::getline(f, s);
			vs.push_back(s);
		}
	}

	for (size_t i = 0; i < vs.size(); i++)
		run(vs[i]);
}
///////////////////////////////////////////////////////////////
vector<string> Script::cleanup(const string &s)
{
	vector<string> vs;

	//remove comments
	// all in lower case

	// split by space or "="
	NetUtil::split(s, vs);
	return vs;
}
///////////////////////////////////////////////////////////////
void Script::run(string sCmd)
{
	vector<string>vs = cleanup(sCmd);
	size_t iSize = vs.size();
	if (vs.size() == 0)
		return;

	//convert to float
	vector<float> vf(5,0.f);
	for(int i=0;i<5;i++)
		if (iSize > i+2)
			vf[i]=strtof(vs[i+2].c_str(),0);

	if (vs[0] == "pause")
		cin.get();

	if (vs[0] == "quit")
		exit(0);

	if ((vs[0] == "print") && iSize > 1)
	{
		for(int i=1;i<iSize;i++)
			cout << vs[i] << " ";
		cout << endl;
	}

	if ((vs[0] == "add") && iSize > 1)
	{
		_net.add(LayerFactory::create(vs[1], vf[0], vf[1], vf[2], vf[3], vf[4]));
	}

	if ((vs[0] == "epochs") && iSize > 1)
	{
		_train.set_epochs((int)(vf[0]));
	}

	if ((vs[0] == "batch_size") && iSize > 1)
	{
		_train.set_batchsize((int)(vf[0]));
	}

	if ((vs[0] == "loss") && iSize > 1)
	{
		_train.set_loss(vs[1]);
	}

	if ((vs[0] == "load") && iSize > 1)
	{
		_data.load(vs[1]);
		_train.set_train_data(_data.train_data(), _data.train_truth());
		_train.set_validation_data(_data.test_data(), _data.test_truth());
	}

	if (vs[0] == "train")
		_train.train();
}
///////////////////////////////////////////////////////////////
