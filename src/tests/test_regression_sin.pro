TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH+=..

SOURCES += test_regression_sin.cpp \
    ../Net.cpp \
    ../NetTrain.cpp \
    ../NetTrainSGD.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../LayerActivation.cpp \
    ../LayerDenseNoBias.cpp \
    ../LayerDenseAndBias.cpp \
    ../LayerDropout.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrain.h \
    ../NetTrainSGD.h \
    ../LayerActivation.h \
    ../LayerDenseNoBias.h \
    ../LayerDenseAndBias.h \
    ../LayerDropout.h

