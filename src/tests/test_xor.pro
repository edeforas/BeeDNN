TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += test_xor.cpp \
    ../Net.cpp \
    ../NetTrainMomentum.cpp \
    ../Layer.cpp \
    ../Activation.cpp \
    ../MatrixUtil.cpp \
    ../ActivationLayer.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrainMomentum.h \
    ../MatrixUtil.h \
    ../ActivationLayer.h
