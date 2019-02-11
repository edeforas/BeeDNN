TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

DEFINES+= "USE_EIGEN_NO"

INCLUDEPATH+=..
INCLUDEPATH+=$$(EIGEN_PATH)

SOURCES += test_regression_sin.cpp \
    ../Net.cpp \
    ../NetTrain.cpp \
    ../NetTrainLearningRate.cpp \
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
    ../NetTrainLearningRate.h \
    ../MatrixUtil.h \
    ../LayerActivation.h \
    ../LayerDenseNoBias.h \
    ../LayerDenseAndBias.h
