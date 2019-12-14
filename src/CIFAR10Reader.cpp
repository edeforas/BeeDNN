/*
	Copyright (c) 2019, Etienne de Foras and the respective contributors
	All rights reserved.

	Use of this source code is governed by a MIT-style license that can be found
	in the LICENSE.txt file.
*/

#include "CIFAR10Reader.h"

#include <fstream>
using namespace std;

// file format and data at: https://www.cs.toronto.edu/~kriz/cifar.html

////////////////////////////////////////////////////////////////////////////////////
bool CIFAR10Reader::read_from_folder(const string& sFolder,MatrixFloat& mRefImages,MatrixFloat& mRefLabels,MatrixFloat& mTestImages,MatrixFloat& mTestLabels)
{
	string sRefImages1=sFolder+"\\data_batch_1.bin";
	string sRefImages2=sFolder+"\\data_batch_2.bin";
	string sRefImages3=sFolder+"\\data_batch_3.bin";
	string sRefImages4=sFolder+"\\data_batch_4.bin";
	string sRefImages5=sFolder+"\\data_batch_5.bin";
	string sTestImages=sFolder+"\\test_batch.bin";

	MatrixFloat mImage, mTruth;
	MatrixFloat mTestImage, mTestTruth;

	//concatenate every train data
	mRefImages.resize(50000, 32 * 32 * 3);
	mRefLabels.resize(50000, 1);

	if(!read_batch(sRefImages1,mImage, mTruth))
		return false;

	copyInto(mImage, mRefImages, 0);
	copyInto(mTruth, mRefLabels, 0);

	if(!read_batch(sRefImages2,mImage, mTruth))
		return false;

	copyInto(mImage, mRefImages, 10000);
	copyInto(mTruth, mRefLabels, 10000);

	if(!read_batch(sRefImages3,mImage, mTruth))
		return false;

	copyInto(mImage, mRefImages, 20000);
	copyInto(mTruth, mRefLabels, 20000);

	if (!read_batch(sRefImages4, mImage, mTruth))
		return false;

	copyInto(mImage, mRefImages, 30000);
	copyInto(mTruth, mRefLabels, 30000);

	if(!read_batch(sRefImages5,mImage, mTruth))
		return false;

	copyInto(mImage, mRefImages, 40000);
	copyInto(mTruth, mRefLabels, 40000);

	//for test, we can write directly into the matrix
	if(!read_batch(sTestImages,mTestImages, mTestLabels))
		return false;

	return true;
}
////////////////////////////////////////////////////////////////////////////////////
bool CIFAR10Reader::read_batch(string sName,MatrixFloat& mData,MatrixFloat& mTruth)
{
	//raw data, 10000 lines, 1st byte is class, next are rgb pixels plane/plane (plane size is 32*32)

	ifstream ifs(sName, ios::binary|ios::in );

	if(!ifs)
		return false;

	mData.resize(10000,32*32*3);
	mTruth.resize(10000,1);

	unsigned char tempLine[32 * 32 *3];

	for(int i=0;i<10000;i++)
	{
		unsigned char ucClass;
		ifs.read((char*)(&ucClass),1);
		mTruth(i)=ucClass;
		
		ifs.read((char*)(tempLine), 32 * 32 *3);
		
		for(int j=0;j< 32 * 32 *3;j++)
			mData(i,j)=tempLine[j];
	}
	return true;
}
////////////////////////////////////////////////////////////////////////////////////
