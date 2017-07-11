TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += test_xor.cpp \
    ../ActivationSigmoid.cpp \
    ../Net.cpp \
    ../DenseLayer.cpp \
    ../Layer.cpp \
    ../Activation.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Activation.h \
    ../ActivationSigmoid.h \
    ../DenseLayer.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h
