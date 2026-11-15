/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#pragma once

#include "Layer.h"
#include "Matrix.h"

#include <vector>

namespace beednn {
class LayerRandomAffine2D : public Layer
{
public:
	LayerRandomAffine2D(Index iRows, Index iCols, Index iChannels,
		float fMaxRotationDegrees, float fMaxTranslationRows, float fMaxTranslationCols);
	virtual ~LayerRandomAffine2D() override;

	virtual Layer* clone() const override;

	virtual void forward(const MatrixFloat& mIn, MatrixFloat& mOut) override;
	virtual void backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn) override;

private:
	struct Transform
	{
		float cosine;
		float sine;
		float translationRows;
		float translationCols;
	};

	void source_position(Index iOutputRow, Index iOutputCol, const Transform& transform,
		float& fSourceRow, float& fSourceCol) const;

	Index _iRows;
	Index _iCols;
	Index _iChannels;
	float _fMaxRotationDegrees;
	float _fMaxTranslationRows;
	float _fMaxTranslationCols;
	std::vector<Transform> _transforms;
};
}
