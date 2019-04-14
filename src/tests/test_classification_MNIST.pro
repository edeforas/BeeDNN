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
    ../LayerSoftmax.cpp \
    ../MNISTReader.cpp \
    ../ConfusionMatrix.cpp \
    ../LayerDense.cpp \
    ../LayerDropout.cpp \
    ../LayerGlobalGain.cpp \
    ../LayerPoolAveraging1D.cpp \
    ../NetTrain.cpp \
	../Loss.cpp \
    ../Optimizer.cpp \
    test_classification_MNIST.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Activation.h \
    ../LayerActivation.h \
    ../LayerSoftmax.h \
    ../Layer.h \
    ../Matrix.h \
    ../Net.h \
    ../MNISTReader.h \
    ../NetTrain.h \
	../Loss.h \
    ../Optimizer.h \
    ../ConfusionMatrix.h \
    ../LayerDense.h \
    ../LayerDropout.h \
    ../LayerPoolAveraging1D.h \
    ../LayerGlobalGain.h
