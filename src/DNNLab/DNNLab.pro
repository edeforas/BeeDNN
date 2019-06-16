QT       += core gui widgets
TARGET = DNNLab
TEMPLATE = app
CONFIG += c++17


#qwt (optional)
#DEFINES+= "USE_QWT"
#INCLUDEPATH += $$(QWT_PATH)/include


#eigen (optional)
DEFINES+= "USE_EIGEN"
INCLUDEPATH += $$(EIGEN_PATH)

#net library
INCLUDEPATH += ..
SOURCES += \
    ../Net.cpp \
    ../NetUtil.cpp \
    ../NetTrain.cpp \
    ../Loss.cpp \
    ../Optimizer.cpp \
    ../Layer.cpp \
    ../Matrix.cpp \
    ../ConfusionMatrix.cpp \
    ../MNISTReader.cpp \
    ../Activation.cpp \
    ../LayerActivation.cpp \
    ../LayerSoftmax.cpp \
    ../LayerDense.cpp \
    ../LayerDropout.cpp \
    ../LayerGaussianNoise.cpp \
    ../LayerGlobalGain.cpp \
    ../LayerPoolAveraging1D.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../ConfusionMatrix.h \
    ../MNISTReader.h \
    ../Net.h \
    ../NetUtil.h \
    ../NetTrain.h \
    ../Loss.h \
    ../Optimizer.h \
    ../LayerActivation.h \
    ../LayerSoftmax.h \
    ../LayerDense.h \
    ../LayerDropout.h \
    ../LayerGaussianNoise.h \
    ../LayerGlobalGain.h \
    ../LayerPoolAveraging1D.h

#GUI
SOURCES += \
    main.cpp\
    mainwindow.cpp \
    SimpleCurveWidget.cpp \
    DataSource.cpp \
    MLEngineBeeDnn.cpp \
    FrameNetwork.cpp \
    FrameNotes.cpp \
    FrameGlobal.cpp \
    FrameLearning.cpp
	
HEADERS  += \
    mainwindow.h \
    SimpleCurveWidget.h \
    DataSource.h \
    MLEngineBeeDnn.h \
    FrameNetwork.h \
    FrameNotes.h \
    FrameGlobal.h \
    FrameLearning.h

FORMS    += \
    mainwindow.ui \
    FrameNotes.ui \
    FrameNetwork.ui \
    FrameGlobal.ui \
    FrameLearning.ui

