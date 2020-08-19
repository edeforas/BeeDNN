#include "DataSourceSelector.h"

#include "NetUtil.h"
#include "MNISTReader.h"
#include "CIFAR10Reader.h"

////////////////////////////////////////////////////////////////////////
void replace_last(string& s, string sOld,string sNew)
{
	auto found = s.rfind(sOld);
	if(found != std::string::npos)
		s.replace(found, sOld.length(), sNew);
}
////////////////////////////////////////////////////////////////////////
DataSourceSelector::DataSourceSelector()
{
	_bHasTrainData=false;
	_bHasTestData=false;
}
////////////////////////////////////////////////////////////////////////
DataSourceSelector::~DataSourceSelector()
{}
////////////////////////////////////////////////////////////////////////
void DataSourceSelector::write(string& s) const
{
	s+=string("DataSource=")+_sName+string("\n");
}
////////////////////////////////////////////////////////////////////////
void DataSourceSelector::read(const string& s)
{
	string sVal=NetUtil::find_key(s,"DataSource");

	load(sVal);
}
////////////////////////////////////////////////////////////////////////
bool DataSourceSelector::load(const string& sName)
{
	if(sName == _sName)
		return true; //already loaded

	clear();
	_sName=sName;

	if (_sName == "")
		return true;
	
	// if has an extension -> custom data text file
	if(_sName.find('.')!=string::npos)
		return load_textfile();

	else if(_sName=="MNIST")
		return load_mnist();

	else if (_sName == "MiniMNIST")
		return load_mini_mnist();

	else if(_sName=="CIFAR10")
		return load_cifar10();
	
	clear();
	return false;
}
////////////////////////////////////////////////////////////////////////
bool DataSourceSelector::load_mnist()
{
	MNISTReader r;
	if(!r.load("."))
		return false;

	_mTrainData = r.train_data();
	_mTrainTruth = r.train_truth();
	_mTestData = r.test_data();
	_mTestTruth = r.test_truth();

	_bHasTrainData=true;
	_bHasTestData=true;

	return true;
}
////////////////////////////////////////////////////////////////////////
bool DataSourceSelector::load_mini_mnist() //MNIST decimated 10x for quick tests
{
	if(!load_mnist())
		return false;

	_mTrainData = decimate(_mTrainData, 10);
	_mTrainTruth = decimate(_mTrainTruth, 10);

	_mTestData = decimate(_mTestData, 10);
	_mTestTruth = decimate(_mTestTruth, 10);

	return true;
}
////////////////////////////////////////////////////////////////////////
bool DataSourceSelector::load_cifar10()
{
	CIFAR10Reader r;
	if(!r.read_from_folder(".",_mTrainData,_mTrainTruth,_mTestData,_mTestTruth))
		return false;

	_mTrainData/=256.f;
	_mTestData/=256.f;

	_bHasTrainData=true;
	_bHasTestData=true;

	return true;
}
////////////////////////////////////////////////////////////////////////
bool DataSourceSelector::load_textfile()
{
	//create 4 file names (may not exist)
	string sTrainData=_sName;
	string sTrainTruth=_sName;
	string sTestData=_sName;
	string sTestTruth=_sName;

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
		_mTestData=fromFile(sTestData);
		_mTestTruth=fromFile(sTestTruth);
	}
	else
	{
		_mTestData.resize(0,0);
		_mTestTruth.resize(0,0);
	}

	_bHasTrainData=(_mTrainData.size()!=0) && (_mTrainTruth.size()!=0) ;
	_bHasTestData=(_mTestData.size()!=0) && (_mTestTruth.size()!=0) ;

	return has_data();
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSourceSelector::train_data() const
{
	return _mTrainData;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSourceSelector::train_truth() const
{
	return _mTrainTruth;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSourceSelector::test_data() const
{
	return _mTestData;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSourceSelector::test_truth() const
{
	return _mTestTruth;
}
////////////////////////////////////////////////////////////////////////
bool DataSourceSelector::has_data() const
{
	return _bHasTrainData || _bHasTestData;
}
////////////////////////////////////////////////////////////////////////
bool DataSourceSelector::has_train_data() const
{
	return _bHasTrainData;
}
////////////////////////////////////////////////////////////////////////
bool DataSourceSelector::has_test_data() const
{
	return _bHasTestData;
}
////////////////////////////////////////////////////////////////////////
int DataSourceSelector::data_size() const
{
	return (int)_mTrainData.cols();
}
////////////////////////////////////////////////////////////////////////
int DataSourceSelector::annotation_cols() const
{
	return (int)_mTrainTruth.cols();
}
////////////////////////////////////////////////////////////////////////
void DataSourceSelector::clear()
{
	_mTrainData.resize(0,0);
	_mTrainTruth.resize(0,0);
	_mTestData.resize(0,0);
	_mTestTruth.resize(0,0);

	_bHasTestData=false;
	_bHasTrainData=false;

	_sName="";
}
////////////////////////////////////////////////////////////////////////
const string DataSourceSelector::name() const
{
	return _sName;
}
////////////////////////////////////////////////////////////////////////
