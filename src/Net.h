/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#ifndef Net_
#define Net_

#include "Matrix.h"
#include <vector>

namespace bee {
class Layer;

class Net
{
public:
    Net();
    virtual ~Net();
    Net& operator=(const Net& other);

	void clear();
	void init();

	// add a layer, take the ownership of the layer
	void add(Layer* l);

	// replace a layer, take the ownership of the layer
	void replace(size_t iLayer,Layer* l);

	const std::vector<Layer*> layers() const;
    Layer& layer(size_t iLayer);
	const Layer& layer(size_t iLayer) const;
	size_t size() const;

	void set_classification_mode(bool bClassificationMode); //true by default
	bool is_classification_mode() const;

	void predict(const MatrixFloat& mIn, MatrixFloat& mOut) const;
	void predict_classes(const MatrixFloat& mIn, MatrixFloat& mClass) const;

    void set_train_mode(bool bTrainMode); // set to true if training, set to false if testing (default)

private:
	bool _bTrainMode;
	std::vector<Layer*> _layers;
	bool _bClassificationMode;
};
}
#endif
