TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += ../Net.cpp \
    ../DenseLayer.cpp \
    ../Layer.cpp \
    ../Activation.cpp \
    test_classification_MNIST.cpp \
    ../MNISTReader.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Activation.h \
    ../DenseLayer.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../MNISTReader.h
