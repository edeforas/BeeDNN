/*
	Copyright (c) 2019, Etienne de Foras and the respective contributors
	All rights reserved.

	Use of this source code is governed by a MIT-style license that can be found
	in the LICENSE.txt file.
*/

#include "CsvFileReader.h"

#include <fstream>
using namespace std;

////////////////////////////////////////////////////////////////////////
void CsvFileReader::replace_last(string& s, const string& sOld, const string& sNew)
{
	auto found = s.rfind(sOld);
	if (found != std::string::npos)
		s.replace(found, sOld.length(), sNew);
}
////////////////////////////////////////////////////////////////////////////////////
bool CsvFileReader::load(const string& sFile)
{
	//create 4 file names (may not exist)
	string sTrainData= sFile;
	string sTrainTruth= sFile;
	string sTestData= sFile;
	string sTestTruth= sFile;

	//create sTrainData
	replace_last(sTrainData,"test","train");
	replace_last(sTrainData,"truth","data");

	//create sTrainTruth
	replace_last(sTrainTruth,"test","train");
	replace_last(sTrainTruth,"data","truth");

	//create sTestData
	replace_last(sTestData,"train","test");
	replace_last(sTestData,"truth","data");

	//create sTestTruth
	replace_last(sTestTruth,"train","test");
	replace_last(sTestTruth,"data","truth");

	//try to load every file
	_mTrainData=fromFile(sTrainData);
	_mTrainTruth=fromFile(sTrainTruth);

	if((sTrainData!=sTestData) && (sTrainTruth!=sTestTruth))
	{
		_mValData=fromFile(sTestData);
		_mValTruth=fromFile(sTestTruth);
	}
	else
	{
		_mValData.resize(0,0);
		_mValTruth.resize(0,0);
	}

	_bHasTrainData=(_mTrainData.size()!=0) && (_mTrainTruth.size()!=0) ;
	_bHasValidationData=(_mValData.size()!=0) && (_mValTruth.size()!=0) ;

	return has_data();
}	
