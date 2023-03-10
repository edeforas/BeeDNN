/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include <sstream>
#include <fstream>
using namespace std;

#include "JsonFile.h"

//////////////////////////////////////////////////////////////////////////////
JsonFile::JsonFile()
{ 
    clear();
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::clear()
{ }
//////////////////////////////////////////////////////////////////////////////
string JsonFile::to_string()
{
	string sOut;

	vector<string> lastIdent;

	for (auto e = _vsItems.begin(); e < _vsItems.end(); e++)
	{
		const auto& vs = *e;
		int iSize = (int)vs.size();
		string sVal = vs[iSize-1];
		string sKey = vs[iSize-2];
		vector<string> newIdent(vs.begin(), vs.end() - 2);


		sOut += sVal + "=" + sKey +"\n";
	}
	return sOut;
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::from_string(const string& s)
{
	_vsItems.clear();
	_vsSectionIndent.clear();
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::enter_section(const string& sSection)
{
	_vsSectionIndent.push_back(sSection);
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::leave_section()
{
	_vsSectionIndent.pop_back();
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add(const string& sKey,const string& sVal)
{
    add_string(sKey, "\"" + sVal + "\"");
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add(const string& sKey, int iVal)
{
    stringstream ss;
    ss << iVal;
    add_string(sKey, ss.str());
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add(const string& sKey, float fVal)
{
    stringstream ss;
    ss << fVal;
    add_string(sKey, ss.str());
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add(const string& sKey, bool bVal)
{
    add_string(sKey, (bVal ? "true" : "false"));
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add_array(const string& sKey, int iSize, const float* pVal)
{
    stringstream ss;
    for (int i = 0; i < iSize; i++)
    {
        ss << pVal[i];
        if (i != iSize - 1)
            ss << ",";
    }

    add_string(sKey, "[" + ss.str() +"]");
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add_string(const string& sKey,const string& sVal)
{
	vector<string> vs = _vsSectionIndent;
	vs.push_back(sKey);
	vs.push_back(sVal);
	_vsItems.push_back(vs);
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::write(const string& sFile)
{
    std::ofstream f(sFile);
    f << this->to_string();
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::read(const string& sFile)
{
	ifstream f(sFile);
	stringstream buf;
	buf << f.rdbuf();

	this->from_string(buf.str());
}
//////////////////////////////////////////////////////////////////////////////