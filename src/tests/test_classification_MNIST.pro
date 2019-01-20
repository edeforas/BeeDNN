TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    ../Net.cpp \
    ../NetTrainMomentum.cpp \
    ../Layer.cpp \
    ../Activation.cpp \
    ../MNISTReader.cpp \
    ../MatrixUtil.cpp \
    ../ConfusionMatrix.cpp \
    ../ActivationLayer.cpp \
    test_classification_MNIST.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../NetTrainMomentum.cpp \
    ../MNISTReader.h \
    ../MatrixUtil.h \
    ../ConfusionMatrix.h \
    ../ActivationLayer.h
