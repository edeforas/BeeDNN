/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "KMeansTrain.h"

#include "KMeans.h"
#include "Matrix.h"

namespace bee {

/////////////////////////////////////////////////////////////////////////////////////////////////
KMeansTrain::KMeansTrain():
	_epochCallBack(nullptr)
{
	_fTrainAccuracy=0.f;
	_fValidationAccuracy=0.f;

    _iEpochs = 100;

	_pKm = nullptr;

    _pmSamplesTrain = nullptr;
    _pmTruthTrain = nullptr;

	_pmSamplesValidation = nullptr;
	_pmTruthValidation = nullptr;

	_iValidationBatchSize = 1024;
	_iBatchSize = 0;
	_bKeepBest = true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
KMeansTrain::~KMeansTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_kmeans(KMeans& km)
{
	_pKm = &km;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
KMeans& KMeansTrain::kmeans()
{
	return *_pKm;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_epochs(int iEpochs) //100 by default
{
    _iEpochs = iEpochs;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
int KMeansTrain::get_epochs() const
{
    return _iEpochs;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_epoch_callback(std::function<void()> epochCallBack)
{
    _epochCallBack = epochCallBack;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_batchsize(Index iBatchSize)
{
	_iBatchSize = iBatchSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
Index KMeansTrain::get_batchsize() const
{
	return _iBatchSize;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_keepbest(bool bKeepBest) //true by default
{
	_bKeepBest = bKeepBest;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
bool KMeansTrain::get_keepbest() const
{
	return _bKeepBest;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
float KMeansTrain::compute_accuracy(const MatrixFloat &mSamples, const MatrixFloat &mTruth) const
{
    Index iNbSamples = mSamples.rows();
	MatrixFloat mOut,mTruthBatch, mSamplesBatch;

	//cut in parts of size _iValidationBatchSize for a lower memory usage
	Index iGood = 0;

	for (Index iStart = 0; iStart < iNbSamples; iStart+=_iValidationBatchSize)
	{
		Index iEnd = iStart + _iValidationBatchSize;
		if (iEnd > iNbSamples)
			iEnd = iNbSamples;
		Index iBatchSize = iEnd - iStart;

		mSamplesBatch = viewRow(mSamples, iStart, iEnd);
		mTruthBatch = viewRow(mTruth, iStart, iEnd);
		
		_pKm->predict_classes(mSamplesBatch, mOut);
			
		for (Index i = 0; i < iBatchSize; i++)
			iGood += mOut(i) == mTruthBatch(i);
	}

	return 100.f*iGood / iNbSamples;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_train_data(const MatrixFloat& mSamples, const MatrixFloat& mTruth)
{
    _pmSamplesTrain = &mSamples;
    _pmTruthTrain = &mTruth;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::set_validation_data(const MatrixFloat& mSamplesValidation, const MatrixFloat& mTruthValidation)
{
	_pmSamplesValidation = &mSamplesValidation;
	_pmTruthValidation = &mTruthValidation;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
void KMeansTrain::fit()
{
	if (_pKm == nullptr)
		return;
	
	_fTrainAccuracy = 0;
	_fValidationAccuracy = 0;

	const MatrixFloat& mSamples = *_pmSamplesTrain;
	const MatrixFloat& mTruth = *_pmTruthTrain;
	Index iNbSamples = mSamples.rows();

	MatrixFloat & mRefVectors = _pKm->ref_vectors();
	MatrixFloat & mRefClasses = _pKm->ref_classes();
	MatrixFloat mRefCentroid, mRefCentroidCount;
	mRefCentroid.setZero(mRefVectors.rows(), mRefVectors.cols());
	mRefCentroidCount.setZero(mRefVectors.rows(), 1);

	//accept batch size == 0 or greater than nb samples  -> full size
	Index iBatchSizeAdjusted = _iBatchSize;
	if ((iBatchSizeAdjusted > iNbSamples) || (iBatchSizeAdjusted == 0))
		iBatchSizeAdjusted = iNbSamples;

	Index iNbRef = mRefVectors.rows();

	float fMaxAccuracy = 0.f;
	KMeans bestKM;
	if (_bKeepBest)
	{
		//need to compte the start accuracy
		if (_pmSamplesValidation == nullptr)
			fMaxAccuracy = compute_accuracy(*_pmSamplesTrain, *_pmTruthTrain);
		else
			fMaxAccuracy = compute_accuracy(*_pmSamplesValidation, *_pmTruthValidation);

		bestKM = *_pKm;
	}

	//init ref vectors, select randomly in full test base
	for (Index i = 0; i < iNbRef; i++)
	{
		Index iPos = randomEngine()() % iNbSamples;

		mRefVectors.row(i) = mSamples.row(iPos);
		mRefClasses.row(i) = mTruth.row(iPos);
	}

	for (int iEpoch = 0; iEpoch < _iEpochs; iEpoch++)
	{
		vector<Index> viPermute = randPerm(iNbSamples);
		Index iNextUpdate = iBatchSizeAdjusted;
		for (Index iS = 0; iS < mSamples.rows(); iS++)
		{
			Index iSample = viPermute[iS];
			int iClass = (int)mTruth(iSample);
			Index iPosBest = -1;
			float fDistBest = 1.e38f;

			for (Index iR = 0; iR < iNbRef; iR++)
			{
				if (mRefClasses(iR) == iClass)
				{
					float fDist = _pKm->compute_dist(mSamples.row(iSample), mRefVectors.row(iR));
					if (fDist < fDistBest)
					{
						fDistBest = fDist;
						iPosBest = iR;
					}
				}
			}

			if (iPosBest != -1)
			{
				// update centroid
				mRefCentroid.row(iPosBest) += mSamples.row(iSample);
				mRefCentroidCount(iPosBest)++;
			}
		
			if ((iS >= iNextUpdate) || (iS == mSamples.rows() - 1))
			{
				//time to update the model
				for (Index iR = 0; iR < iNbRef; iR++)
				{
					if (mRefCentroidCount(iR) != 0)
						mRefVectors.row(iR) = mRefCentroid.row(iR) / mRefCentroidCount(iR);
				}

				mRefCentroid.setZero();
				mRefCentroidCount.setZero();

				iNextUpdate += iBatchSizeAdjusted;
			}
		}

		_fTrainAccuracy = compute_accuracy(mSamples, mTruth);
		float fSelectedAccuracy = _fTrainAccuracy;

		if (_pmSamplesValidation != nullptr)
		{
			_fValidationAccuracy = compute_accuracy(*_pmSamplesValidation, *_pmTruthValidation);
			fSelectedAccuracy = _fValidationAccuracy;
		}

		if (_bKeepBest)
		{
			if (fMaxAccuracy < fSelectedAccuracy)
			{
				fMaxAccuracy = fSelectedAccuracy;
				bestKM = *_pKm;
			}
		}

		if (_epochCallBack)
			_epochCallBack();
	}

	if (_bKeepBest)
		(*_pKm).operator=(bestKM);
}
/////////////////////////////////////////////////////////////////////////////////////////////
float KMeansTrain::get_current_validation_accuracy() const
{
	return _fValidationAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
float KMeansTrain::get_current_train_accuracy() const
{
	return _fTrainAccuracy;
}
/////////////////////////////////////////////////////////////////////////////////////////////
}