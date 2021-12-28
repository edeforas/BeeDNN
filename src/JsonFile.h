/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef JsonFile_
#define JsonFile_

#include <string>
#include <vector>
#include <map>
using namespace std;

class JsonFile {
public:
    JsonFile();
    void clear();
    string to_string();

    void enter_section(string sSection);
    void leave_section();

    void add_key(string sKey, string sVal);
    void add_key(string sKey, int iVal);
    void add_key(string sKey, float fVal);
    void add_key(string sKey, bool bVal);

private:
	bool _bPendingComma;
    void add(string sKey, string sValNoFormatting);

    string _sSectionIndent;

    //   string find_key(string s, string sKey);
 //   void split(string s, vector<string>& vsItems, char cDelimiter=' ');
    //map<string, string> _allPairs;

    string _sOut;
};

#endif
