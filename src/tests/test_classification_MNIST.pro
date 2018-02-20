TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += ../Net.cpp \
    ../Layer.cpp \
    ../Activation.cpp \
    test_classification_MNIST.cpp \
    ../MNISTReader.cpp \
    ../MatrixUtil.cpp \
    ../ConfusionMatrix.cpp \
    ../ActivationLayer.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../MNISTReader.h \
    ../MatrixUtil.h \
    ../ConfusionMatrix.h \
    ../ActivationLayer.h
