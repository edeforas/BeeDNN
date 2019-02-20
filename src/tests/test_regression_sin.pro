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
    ../MatrixUtil.cpp \
    ../LayerActivation.cpp \
    ../LayerDenseNoBias.cpp \
    ../LayerDenseAndBias.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrain.h \
    ../NetTrainSGD.h \
    ../MatrixUtil.h \
    ../LayerActivation.h \
    ../LayerDenseNoBias.h \
    ../LayerDenseAndBias.h
