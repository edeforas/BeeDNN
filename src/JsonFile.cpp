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
{
    _sSectionIndent = "    ";
    _sOut = "";
    _bPendingComma = false;
}
//////////////////////////////////////////////////////////////////////////////
string JsonFile::to_string()
{
    return "{"+_sOut+"\n}";
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::enter_section(string sSection)
{
    if (_bPendingComma)
        _sOut += ",";

    _sOut += "\n" + _sSectionIndent + "\"" + sSection + "\":{";

    _sSectionIndent += "    ";
	_bPendingComma=false;
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::leave_section()
{
    _sSectionIndent = _sSectionIndent.substr(4);

    _sOut += "\n" + _sSectionIndent + "}";
	_bPendingComma=false;
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add_key(string sKey, string sVal)
{
    add(sKey, "\"" + sVal + "\"");
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add_key(string sKey, int iVal)
{
    stringstream ss;
    ss << iVal;
    add(sKey, ss.str());
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add_key(string sKey, float fVal)
{
    stringstream ss;
    ss << fVal;
    add(sKey, ss.str());
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add_key(string sKey, bool bVal)
{
    add(sKey, (bVal ? "true" : "false"));
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add(string sKey, string sValNoFormatting)
{
    if (_bPendingComma)
        _sOut += ",";
	
    _sOut += "\n" + _sSectionIndent + "\"" + sKey + "\":" + sValNoFormatting;
	_bPendingComma=true;
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::save(string sFile)
{
    std::ofstream f(sFile);
    f << this->to_string();
}
//////////////////////////////////////////////////////////////////////////////


/*string JsonFile::find_key(string s, string sKey)
{
    auto i = s.find(sKey + ":");

    if (i == string::npos)
        return "";

    i += sKey.size() + 1;

    auto i2 = s.find("\n", i);

    if (i2 == string::npos)
        i2 = s.size();

    string s2 = s.substr(i, i2 - i);

    //trim right
    auto i3 = s2.find_last_not_of(" \t\r\n");
    if (i3 != string::npos)
        return s2.substr(0, i3 + 1);
    else
        return s2;
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::split(string s, vector<string>& vsItems, char cDelimiter)
{
    vsItems.clear();

    istringstream f(s);
    string sitem;
    while (getline(f, sitem, cDelimiter))
        vsItems.push_back(sitem);
}
//////////////////////////////////////////////////////////////////////////////
*/