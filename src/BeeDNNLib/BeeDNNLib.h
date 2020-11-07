#ifndef _BeeDNNLib_
#define _BeeDNNLib_


#ifdef _WIN32
	#ifdef BEEDNNLIB_BUILD
		#define BEEDNN_EXPORT extern "C" __declspec(dllexport)
	#else
		#define BEEDNN_EXPORT extern "C" __declspec(dllimport)
	#endif
#endif

BEEDNN_EXPORT void hello();
BEEDNN_EXPORT float activation(char * activ, float f);

#endif
