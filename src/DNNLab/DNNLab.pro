QT       += core gui widgets
TARGET = DNNLab
TEMPLATE = app
CONFIG += c++17

#qwt (optional)
#DEFINES+= "USE_QWT"
#INCLUDEPATH += $$(QWT_PATH)/include
#LIBS += -L$$(QWT_PATH)\lib -lqwt
#DEFINES += QWT_DLL

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
    ../LayerGaussianDropout.cpp \
    ../LayerGaussianNoise.cpp \
    ../LayerGlobalGain.cpp \
    ../LayerGlobalBias.cpp \
    ../LayerPRelu.cpp \
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
    ../LayerGaussianDropout.h \
    ../LayerGaussianNoise.h \
    ../LayerGlobalGain.h \
    ../LayerGlobalBias.h \
    ../LayerPRelu.h \
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

RESOURCES += BeeDNN.qrc
RC_ICONS = BeeDNN.ico
