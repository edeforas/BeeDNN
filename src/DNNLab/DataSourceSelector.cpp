#include "DataSourceSelector.h"

#include "NetUtil.h"
#include "MNISTReader.h"
#include "CIFAR10Reader.h"
#include "CsvFileReader.h"

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
	if (!r.load("."))
		return false;

	_mTrainData = r.train_data();
	_mTrainTruth = r.train_truth();
	_mTestData = r.test_data();
	_mTestTruth = r.test_truth();

	_bHasTrainData = true;
	_bHasTestData = true;

	return true;
}
////////////////////////////////////////////////////////////////////////
bool DataSourceSelector::load_textfile()
{
	CsvFileReader r;
	if (!r.load(_sName))
		return false;

	_mTrainData = r.train_data();
	_mTrainTruth = r.train_truth();
	_mTestData = r.test_data();
	_mTestTruth = r.test_truth();

	_bHasTrainData = r.has_train_data();
	_bHasTestData = r.has_test_data();

	return true;
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
