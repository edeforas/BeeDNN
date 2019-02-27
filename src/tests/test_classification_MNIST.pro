TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

#eigen (optional)
DEFINES+= "USE_EIGEN"
INCLUDEPATH += $$(EIGEN_PATH)

SOURCES += \
    ../Net.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../Activation.cpp \
    ../LayerActivation.cpp \
    ../MNISTReader.cpp \
    ../MatrixUtil.cpp \
    ../ConfusionMatrix.cpp \
    ../LayerDenseAndBias.cpp \
    ../LayerDenseNoBias.cpp \
    ../LayerDropout.cpp \
    ../NetTrain.cpp \
    ../NetTrainSGD.cpp \
    test_classification_MNIST.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Activation.h \
    ../layerActivation.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../MNISTReader.h \
    ../NetTrain.h \
    ../NetTrainSGD.h \
    ../MatrixUtil.h \
    ../ConfusionMatrix.h \
    ../LayerDenseNoBias.h \
    ../LayerDenseAndBias.h \
	../LayerDropout.h
