TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH+=..

SOURCES += test_xor.cpp \
    ../Net.cpp \
    ../NetTrain.cpp \
    ../NetTrainSGD.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../LayerActivation.cpp \
    ../LayerDenseAndBias.cpp \
    ../LayerDenseNoBias.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrain.h \
    ../NetTrainSGD.h \
    ../MatrixUtil.h \
    ../LayerActivation.h \
    ../LayerDenseAndBias.h \
    ../LayerDenseNoBias.h
