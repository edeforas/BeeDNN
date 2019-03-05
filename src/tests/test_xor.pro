TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH+=..

SOURCES += test_xor.cpp \
    ../Net.cpp \
    ../Optimizer.cpp \
    ../NetTrain.cpp \
    ../NetTrainSGD.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../LayerActivation.cpp \
    ../LayerDense.cpp \
    ../LayerDropout.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../Optimizer.h \
    ../NetTrain.h \
    ../NetTrainSGD.h \
    ../LayerActivation.h \
    ../LayerDense.h \
    ../LayerDropout.h
