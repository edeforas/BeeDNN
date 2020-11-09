#ifndef _BeeDNNLib_
#define _BeeDNNLib_

#ifdef _WIN32
	#ifdef BEEDNNLIB_BUILD
		#define BEEDNN_EXPORT extern "C" __declspec(dllexport)
	#else
		#define BEEDNN_EXPORT extern "C" __declspec(dllimport)
	#endif
#endif

BEEDNN_EXPORT float activation(char * activ, float f);

BEEDNN_EXPORT void * create();
BEEDNN_EXPORT void set_classification_mode(void* pNN, int32_t _iClassificationMode);
BEEDNN_EXPORT void add_layer(void* pNN, char *layer);
BEEDNN_EXPORT void predict(void* pNN, const float *pIn, float *pOut);

#endif
