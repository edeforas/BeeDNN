/*
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT-style license that can be found
    in the LICENSE.txt file.
*/

#include "LayerRandomAffine2D.h"

#include <cassert>
#include <cmath>
#include <random>

using namespace std;
namespace beednn {
namespace {

float bilinear_sample(const float* pImage, Index iRows, Index iCols, float fRow, float fCol)
{
	const Index row0 = (Index)floorf(fRow);
	const Index col0 = (Index)floorf(fCol);
	const float rowFraction = fRow - (float)row0;
	const float colFraction = fCol - (float)col0;
	float value = 0.f;

	for (Index rowOffset = 0; rowOffset <= 1; rowOffset++)
	{
		const Index row = row0 + rowOffset;
		if (row < 0 || row >= iRows)
			continue;
		const float rowWeight = rowOffset == 0 ? 1.f - rowFraction : rowFraction;

		for (Index colOffset = 0; colOffset <= 1; colOffset++)
		{
			const Index col = col0 + colOffset;
			if (col < 0 || col >= iCols)
				continue;
			const float colWeight = colOffset == 0 ? 1.f - colFraction : colFraction;
			value += pImage[row * iCols + col] * rowWeight * colWeight;
		}
	}

	return value;
}

void scatter_bilinear_gradient(float* pGradient, Index iRows, Index iCols, float fRow, float fCol, float fGradient)
{
	const Index row0 = (Index)floorf(fRow);
	const Index col0 = (Index)floorf(fCol);
	const float rowFraction = fRow - (float)row0;
	const float colFraction = fCol - (float)col0;

	for (Index rowOffset = 0; rowOffset <= 1; rowOffset++)
	{
		const Index row = row0 + rowOffset;
		if (row < 0 || row >= iRows)
			continue;
		const float rowWeight = rowOffset == 0 ? 1.f - rowFraction : rowFraction;

		for (Index colOffset = 0; colOffset <= 1; colOffset++)
		{
			const Index col = col0 + colOffset;
			if (col < 0 || col >= iCols)
				continue;
			const float colWeight = colOffset == 0 ? 1.f - colFraction : colFraction;
			pGradient[row * iCols + col] += fGradient * rowWeight * colWeight;
		}
	}
}
}

LayerRandomAffine2D::LayerRandomAffine2D(Index iRows, Index iCols, Index iChannels,
	float fMaxRotationDegrees, float fMaxTranslationRows, float fMaxTranslationCols) :
	Layer("RandomAffine2D"),
	_iRows(iRows),
	_iCols(iCols),
	_iChannels(iChannels),
	_fMaxRotationDegrees(fMaxRotationDegrees),
	_fMaxTranslationRows(fMaxTranslationRows),
	_fMaxTranslationCols(fMaxTranslationCols)
{
	assert(_iRows > 0);
	assert(_iCols > 0);
	assert(_iChannels > 0);
	assert(_fMaxRotationDegrees >= 0.f);
	assert(_fMaxTranslationRows >= 0.f);
	assert(_fMaxTranslationCols >= 0.f);
}

LayerRandomAffine2D::~LayerRandomAffine2D()
{ }

Layer* LayerRandomAffine2D::clone() const
{
	return new LayerRandomAffine2D(_iRows, _iCols, _iChannels,
		_fMaxRotationDegrees, _fMaxTranslationRows, _fMaxTranslationCols);
}

void LayerRandomAffine2D::source_position(Index iOutputRow, Index iOutputCol, const Transform& transform,
	float& fSourceRow, float& fSourceCol) const
{
	const float centerRow = ((float)_iRows - 1.f) * 0.5f;
	const float centerCol = ((float)_iCols - 1.f) * 0.5f;
	const float relativeRow = (float)iOutputRow - centerRow - transform.translationRows;
	const float relativeCol = (float)iOutputCol - centerCol - transform.translationCols;

	fSourceRow = transform.cosine * relativeRow + transform.sine * relativeCol + centerRow;
	fSourceCol = -transform.sine * relativeRow + transform.cosine * relativeCol + centerCol;
}

void LayerRandomAffine2D::forward(const MatrixFloat& mIn, MatrixFloat& mOut)
{
	const Index planeSize = _iRows * _iCols;
	assert(mIn.cols() == planeSize * _iChannels);
	mOut.resizeLike(mIn);
	_transforms.resize(mIn.rows());

	const float degreesToRadians = 0.017453292519943295f;
	uniform_real_distribution<float> rotationDistribution(-_fMaxRotationDegrees, _fMaxRotationDegrees);
	uniform_real_distribution<float> rowTranslationDistribution(-_fMaxTranslationRows, _fMaxTranslationRows);
	uniform_real_distribution<float> colTranslationDistribution(-_fMaxTranslationCols, _fMaxTranslationCols);
	default_random_engine& engine = randomEngine();

	for (Index sample = 0; sample < mIn.rows(); sample++)
	{
		Transform& transform = _transforms[sample];
		if (_bTrainMode)
		{
			const float angle = rotationDistribution(engine) * degreesToRadians;
			transform.cosine = cosf(angle);
			transform.sine = sinf(angle);
			transform.translationRows = rowTranslationDistribution(engine);
			transform.translationCols = colTranslationDistribution(engine);
		}
		else
		{
			transform.cosine = 1.f;
			transform.sine = 0.f;
			transform.translationRows = 0.f;
			transform.translationCols = 0.f;
		}

		for (Index channel = 0; channel < _iChannels; channel++)
		{
			const float* pIn = mIn.row(sample).data() + channel * planeSize;
			float* pOut = mOut.row(sample).data() + channel * planeSize;
			for (Index row = 0; row < _iRows; row++)
			{
				for (Index col = 0; col < _iCols; col++)
				{
					float sourceRow;
					float sourceCol;
					source_position(row, col, transform, sourceRow, sourceCol);
					pOut[row * _iCols + col] = bilinear_sample(pIn, _iRows, _iCols, sourceRow, sourceCol);
				}
			}
		}
	}
}

void LayerRandomAffine2D::backpropagation(const MatrixFloat& mIn, const MatrixFloat& mGradientOut, MatrixFloat& mGradientIn)
{
	const Index planeSize = _iRows * _iCols;
	assert(mIn.cols() == planeSize * _iChannels);
	assert(mGradientOut.rows() == mIn.rows());
	assert(mGradientOut.cols() == mIn.cols());
	assert(_transforms.size() == (size_t)mIn.rows());

	if (_bFirstLayer)
		return;

	mGradientIn.setZero(mIn.rows(), mIn.cols());
	for (Index sample = 0; sample < mIn.rows(); sample++)
	{
		const Transform& transform = _transforms[sample];
		for (Index channel = 0; channel < _iChannels; channel++)
		{
			float* pGradientIn = mGradientIn.row(sample).data() + channel * planeSize;
			const float* pGradientOut = mGradientOut.row(sample).data() + channel * planeSize;
			for (Index row = 0; row < _iRows; row++)
			{
				for (Index col = 0; col < _iCols; col++)
				{
					float sourceRow;
					float sourceCol;
					source_position(row, col, transform, sourceRow, sourceCol);
					scatter_bilinear_gradient(pGradientIn, _iRows, _iCols, sourceRow, sourceCol,
						pGradientOut[row * _iCols + col]);
				}
			}
		}
	}
}
}
