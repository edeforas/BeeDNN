#include "Script.h"

#include <fstream>
#include <iostream>
#include <stdlib.h> // for strtof

#include "NetUtil.h"
#include "LayerFactory.h"

///////////////////////////////////////////////////////////////
Script::Script()
{}
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
		if (iSize > i+1)
			vf[i]=strtof(vs[i+1].c_str(),0);

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



}
///////////////////////////////////////////////////////////////
