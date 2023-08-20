/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef JsonFile_
#define JsonFile_

#include <string>

class JsonFileWriter {
public:
    JsonFileWriter();

    void clear();
    std::string to_string();
    void save(const std::string& sFile);

    void enter_section(const std::string& sSection);
    void leave_section();

    void add(const std::string& sKey, const std::string& sVal);
    void add(const std::string& sKey, int iVal);
    void add(const std::string& sKey, float fVal);
    void add(const std::string& sKey, bool bVal);

    void add_array(const std::string& sKey, int iSize, const float* pVal);

private:
    void add_string(const std::string& sKey, const std::string& s);
    
    bool _bPendingComma;
    std::string _sSectionIndent;
    std::string _sOut;
};

#endif
