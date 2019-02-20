TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    ../Net.cpp \
    ../Layer.cpp \
    ../Activation.cpp \
    ../MNISTReader.cpp \
    ../MatrixUtil.cpp \
    ../ConfusionMatrix.cpp \
    ../LayerDenseAndBias.cpp \
    ../NetTrain.cpp \
    ../NetTrainSGD.cpp \
    test_classification_MNIST.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../MNISTReader.h \
    ../NetTrain.h \
    ../NetTrainSGD.h \
    ../MatrixUtil.h \
    ../ConfusionMatrix.h \
    ../LayerDenseAndBias.h
