/*
    Copyright (c) 2020, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "NetTrainHogwild.h"

#include "Net.h"
#include "Layer.h"
#include "Matrix.h"
#include "Loss.h"
#include "Regularizer.h"
#include "Optimizer.h"

#include <cmath>
#include <cassert>
#include <omp.h>

/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainHogwild::NetTrainHogwild(): NetTrain()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
NetTrainHogwild::~NetTrainHogwild()
{ }
/////////////////////////////////////////////////////////////////////////////////////////////////
void NetTrainHogwild::train_one_epoch( const MatrixFloat& mSampleShuffled, const MatrixFloat& mTruthShuffled)
{
	Index iNbThread = omp_get_max_threads();
	Index iNbSamples = mSampleShuffled.rows();
	
	Net& netShare = *_pNet;
	vector<Net> vNet(iNbThread);
	for (int i = 0; i < iNbThread; i++)
	{
		vNet[i] = netShare;
		vNet[i].set_train_mode(true); //todo remove
	}
	vector<NetTrain> vNetTrain(iNbThread);
	for (Index i = 0; i < iNbThread; i++)
	{
		vNetTrain[i] = *this;
		vNetTrain[i].set_net(vNet[i]);
	}

	//compute nb batches taking into account the last smaller one
	Index iNbBatches = iNbSamples / _iBatchSizeAdjusted;
//	if (iNbBatches*_iBatchSizeAdjusted < iNbSamples)
//		iNbBatches++;

	
#pragma omp parallel for
	for(Index iBatch=0;iBatch<iNbBatches;iBatch++)
	{
		int iThread= omp_get_thread_num();

		// compute mini batches range
		Index iBatchStart = iBatch* _iBatchSizeAdjusted;
		Index iBatchEnd = iBatchStart + _iBatchSizeAdjusted;
	/*	if (iBatchEnd > iNbSamples)
			iBatchEnd = iNbSamples;
		*/
		const MatrixFloat mSample = viewRow(mSampleShuffled, iBatchStart, iBatchEnd);
		const MatrixFloat mTruth = viewRow(mTruthShuffled, iBatchStart, iBatchEnd);

		NetTrain& nt = vNetTrain[iThread];
		Net& n = nt.net();

		for (size_t i = 0; i < n.size(); i++)
		{
			n.layer(i).weights() = netShare.layer(i).weights();
			n.layer(i).bias() = netShare.layer(i).bias();
		}

		nt.train_batch(mSample, mTruth);

		for (size_t i = 0; i < n.size(); i++)
		{
			netShare.layer(i).weights() = n.layer(i).weights();
			netShare.layer(i).bias() = n.layer(i).bias();
		}
					
		iBatchStart = iBatchEnd;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////
