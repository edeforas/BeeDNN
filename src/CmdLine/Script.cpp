#include "Script.h"

#include <fstream>
#include <iostream>

#include "NetUtil.h"

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
	if (vs.size() == 0)
		return;

	if (vs[0] == "pause")
		cin.get();

	if (vs[0] == "quit")
		exit(0);

	if ((vs[0] == "print") && vs.size() > 1)
	{
		cout << vs[1] << endl;
	}

}
///////////////////////////////////////////////////////////////
