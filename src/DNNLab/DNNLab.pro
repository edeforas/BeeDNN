QT       += core gui widgets
TEMPLATE = app
CONFIG += c++17

#qwt (optional)
DEFINES+= "USE_QWT"
INCLUDEPATH += $$(QWT_PATH)/include
DEFINES += QWT_DLL


CONFIG( debug, debug|release ) {
	TARGET = DNNLab_debug
	LIBS += -L$$(QWT_PATH)\lib -lqwtd
} else {
	TARGET = DNNLab
	LIBS += -L$$(QWT_PATH)\lib -lqwt
}




#eigen (optional)
DEFINES+= "USE_EIGEN"
INCLUDEPATH += $$(EIGEN_PATH)

#net library
INCLUDEPATH += ..
SOURCES += \
    ../LayerBias.cpp \
    ../LayerPoolMax1D.cpp \
    ../LayerUniformNoise.cpp \
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
    ../LayerPoolMax2D.cpp
    ../LayerPoolAveraging1D.cpp

HEADERS += \
    ../Activation.h \
    ../Layer.h \
    ../LayerBias.h \
    ../LayerPoolMax1D.h \
    ../LayerUniformNoise.h \
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
    ../LayerPoolMax2D.h
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

