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
JsonFile::JsonFile():
    _iNbSections(0)
{ 
    clear();
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::clear()
{
   // _allPairs.clear();
    _sSection = "";
    _iNbSections = 0;
    _sSectionIndent = "";
    _sOut = "";
    _sSeparator = "\n";

    enter_section("");
}
//////////////////////////////////////////////////////////////////////////////
string JsonFile::to_string()
{
    /*
    string s;
    for (auto it = _allPairs.begin(); it != _allPairs.end(); it++)
    {
        s = s + it->first + ": " + it->second + "\n";
    }
    */

    leave_section();
    return _sOut;
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::enter_section(string sSection)
{
    _sSection = sSection;
    _iNbSections++;

    if(sSection.empty())
        _sOut += _sSectionIndent + "{";
    else
        _sOut += "\n" + _sSectionIndent + "\"" + sSection + "\":{";

    _sSectionIndent = "";
    for (int i = 0; i < _iNbSections; i++)
        _sSectionIndent += "    ";
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::leave_section()
{
    _iNbSections--;

    _sSectionIndent = "";
    for (int i = 0; i < _iNbSections; i++)
        _sSectionIndent += "    ";

    _sOut += "\n" + _sSectionIndent + "}";
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add_key(string sKey, string sVal)
{
    _sOut += "\n"+_sSectionIndent + "\"" +sKey + "\":\"" + sVal + "\"";
//    _allPairs.insert({ sKey,sVal });
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add_key(string sKey, int iVal)
{
    stringstream ss;
    ss << iVal;
    _sOut += "\n"+_sSectionIndent + "\"" +sKey + +"\":" + ss.str();
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add_key(string sKey, float fVal)
{
    stringstream ss;
    ss << fVal;
    _sOut += "\n"+_sSectionIndent + "\"" + sKey + "\":" + ss.str();
}
//////////////////////////////////////////////////////////////////////////////
void JsonFile::add_key(string sKey, bool bVal)
{
    _sOut += "\n"+_sSectionIndent + "\"" + sKey + "\":" + (bVal ? "true" : "false");
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