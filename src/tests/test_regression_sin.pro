TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

#eigen use (optional)
DEFINES+= "USE_EIGEN"
INCLUDEPATH+=$$(EIGEN_PATH)

INCLUDEPATH+=..

SOURCES += test_regression_sin.cpp \
    ../Net.cpp \
    ../NetTrain.cpp \
    ../Loss.cpp \
    ../Optimizer.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../LayerActivation.cpp \
    ../LayerDense.cpp \
    ../LayerDropout.cpp \
    ../LayerPoolAveraging1D.cpp \
    ../LayerGlobalGain.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrain.h \
    ../Loss.h \
    ../Optimizer.h \
    ../LayerActivation.h \
    ../LayerDense.h \
    ../LayerDropout.h \
    ../LayerPoolAveraging1D.h \
    ../LayerGlobalGain.h
