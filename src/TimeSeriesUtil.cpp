#include "TimeSeriesUtil.h"

////////////////////////////////////////////////////////////////////////////////
void TimeSeriesUtil::generate_windowed_data(const MatrixFloat & mIn, int iWindowSize, MatrixFloat & mWindowed) //no strides for now
{
	Index iNbRowsIn = mIn.rows();
	Index iNbColsIn = mIn.cols();

	Index iNbRowsOut = mIn.rows()- iWindowSize;
	Index iNbColsOut = mIn.cols()*iWindowSize;

	mWindowed.resize(iNbRowsOut, iNbColsOut);

	for(Index iR= 0;iR< iNbRowsOut;iR++)
		for (Index iWS = 0; iWS< iWindowSize;iWS++)
			for (Index iC = 0; iC < iNbColsIn; iC++)
				mWindowed(iR, iWS*iNbColsIn+iC) = mIn(iR+iWS, iC);
}
////////////////////////////////////////////////////////////////////////////////