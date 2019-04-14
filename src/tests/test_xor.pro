TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

#eigen use (optional)
#DEFINES+= "USE_EIGEN"
#INCLUDEPATH+=$$(EIGEN_PATH)

INCLUDEPATH+=..

SOURCES += test_xor.cpp \
    ../Net.cpp \
    ../Optimizer.cpp \
    ../NetTrain.cpp \
	../Loss.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../LayerActivation.cpp \
    ../LayerSoftmax.cpp \
    ../LayerDense.cpp \
    ../LayerDropout.cpp \
    ../LayerPoolAveraging1D.cpp \
    ../LayerGlobalGain.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../Optimizer.h \
    ../NetTrain.h \
	../Loss.h \
    ../LayerActivation.h \
    ../LayerSoftmax.cpp \
    ../LayerDense.h \
    ../LayerPoolAveraging1D.h \
    ../LayerGlobalGain.h \
    ../LayerDropout.h
