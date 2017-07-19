TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += test_sin.cpp \
    ../ActivationSigmoid.cpp \
    ../Net.cpp \
    ../DenseLayer.cpp \
    ../Layer.cpp \
    ../Activation.cpp \
    ../ActivationRelu.cpp \
    ../ActivationTanh.cpp \
    ../ActivationSelu.cpp \
    ../ActivationLinear.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Activation.h \
    ../ActivationSigmoid.h \
    ../DenseLayer.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../ActivationRelu.h \
    ../ActivationTanh.h \
    ../ActivationSelu.h \
    ../ActivationLinear.h
