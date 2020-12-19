#include "DataSource.h"
/*
////////////////////////////////////////////////////////////////////////
void replace_last(string& s, string sOld,string sNew)
{
	auto found = s.rfind(sOld);
	if(found != std::string::npos)
		s.replace(found, sOld.length(), sNew);
}
*/
////////////////////////////////////////////////////////////////////////
DataSource::DataSource()
{
	_bHasTrainData=false;
	_bHasValidationData=false;
}
////////////////////////////////////////////////////////////////////////
DataSource::~DataSource()
{ }
////////////////////////////////////////////////////////////////////////

/*bool DataSource::load_mini_mnist() //MNIST decimated 10x for quick tests
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
/*bool DataSource::load_cifar10()
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
/*
////////////////////////////////////////////////////////////////////////
bool DataSource::load_textfile()
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
*/
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::train_data() const
{
	return _mTrainData;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::train_truth() const
{
	return _mTrainTruth;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::validation_data() const
{
	return _mValData;
}
////////////////////////////////////////////////////////////////////////
const MatrixFloat& DataSource::validation_truth() const
{
	return _mValTruth;
}
////////////////////////////////////////////////////////////////////////
bool DataSource::has_data() const
{
	return _bHasTrainData || _bHasValidationData;
}
////////////////////////////////////////////////////////////////////////
bool DataSource::has_train_data() const
{
	return _bHasTrainData;
}
////////////////////////////////////////////////////////////////////////
bool DataSource::has_validation_data() const
{
	return _bHasValidationData;
}
////////////////////////////////////////////////////////////////////////
int DataSource::data_size() const
{
	return (int)_mTrainData.cols();
}
////////////////////////////////////////////////////////////////////////
int DataSource::annotation_cols() const
{
	return (int)_mTrainTruth.cols();
}
////////////////////////////////////////////////////////////////////////
