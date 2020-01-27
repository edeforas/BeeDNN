/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Net_
#define Net_

#include <vector>
#include <string>
using namespace std;

class Layer;
#include "Matrix.h"

class Net
{
public:
    Net();
    virtual ~Net();
    Net& operator=(const Net& other);

	void clear();
	void init();

    void add_dense_layer(Index inSize, Index outSize, bool bHasBias=true);
	void add_activation_layer(string sType);
	void add_prelu_layer();
	void add_softmax_layer();
    void add_dropout_layer(float fRatio);
    void add_gaussian_dropout_layer(float fProba);
	
	void add_uniform_noise_layer(float fNoise); 
	void add_gaussian_noise_layer(float fStd);
    
	void add_globalgain_layer();
	void add_gain_layer();

	void add_globalbias_layer();
	void add_bias_layer();
	
	void add_poolmax2D_layer(Index iInRow, Index iInCols, Index iInChannels, Index iRowFactor = 2, Index iColFactor = 2);
	void add_convolution2D_layer(Index iInRows, Index iInCols, Index iInChannels, Index iKernelRows, Index iKernelCols, Index iOutChannels, Index iRowStride=1, Index iColStride=1);
	void add_channel_bias_layer(Index iInRows, Index iInCols, Index iInChannels);

    void set_input_size(int iInputSize);
    int input_size() const;

    const vector<Layer*> layers() const;
    Layer& layer(size_t iLayer);
	const Layer& layer(size_t iLayer) const;
	size_t size() const;

	void set_classification_mode(bool bClassificationMode); //true by default
	bool is_classification_mode() const;

	void forward(const MatrixFloat& mIn, MatrixFloat& mOut) const;
	void classify(const MatrixFloat& mIn, MatrixFloat& mClass) const;

    void set_train_mode(bool bTrainMode); // set to true if training, set to false if testing (default)

private:
	bool _bTrainMode;
	vector<Layer*> _layers;
    int _iInputSize;
	bool _bClassificationMode;
};

#endif
