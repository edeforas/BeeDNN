/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

// Hogwild distributed learning over mini batches, as in:
// https://arxiv.org/pdf/1106.5730.pdf

#ifndef NetTrainHogwild_
#define NetTrainHogwild_

#include "NetTrain.h"

class NetTrainHogwild: public NetTrain
{
public:
    NetTrainHogwild();
    virtual ~NetTrainHogwild();

protected:
	virtual void train_one_epoch(const MatrixFloat& mSampleShuffled, const MatrixFloat& mTruthShuffled) override;

};

#endif
