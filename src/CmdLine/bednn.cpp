#include <iostream>
#include <chrono>
#include <vector>
#include <string>
using namespace std;

#include "NetScript.h"

/////////////////////////////////////////////////////////////////
void cmd_parser(int argc, char *argv[],vector<string>& vsArgs)
{
	vsArgs.clear();
	for (int i = 0; i < argc; i++)
		vsArgs.push_back(argv[i]);

#ifdef _WIN64
	vsArgs.erase(vsArgs.begin()); //remove exe name under win32/64
#endif
}
/////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{	
	vector<string> vsArgs;
	cmd_parser(argc, argv, vsArgs);

	if (vsArgs.size() == 1)
	{
		NetScript cmd;
		cmd.run_file(vsArgs[0]);
	}

	return 0;
}
/////////////////////////////////////////////////////////////////
