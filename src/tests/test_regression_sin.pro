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
    ../Optimizer.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../LayerActivation.cpp \
    ../LayerDense.cpp \
    ../LayerDropout.cpp \
    ../LayerPoolAverage1D.cpp \
    ../LayerGlobalGain.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrain.h \
    ../Optimizer.h \
    ../LayerActivation.h \
    ../LayerDense.h \
    ../LayerDropout.h \
    ../LayerPoolAverage1D.h \
    ../LayerGlobalGain.h
