/*
	Copyright (c) 2019, Etienne de Foras and the respective contributors
	All rights reserved.

	Use of this source code is governed by a MIT-style license that can be found
	in the LICENSE.txt file.
*/

// LayerGTU as in : https://arxiv.org/pdf/1612.08083.pdf

#include "LayerGTU.h"
namespace bee {
///////////////////////////////////////////////////////////////////////////////
LayerGTU::LayerGTU() :
	LayerGatedActivation("Tanh", "Sigmoid")
{ }
///////////////////////////////////////////////////////////////////////////////
}