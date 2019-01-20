QT       += core gui widgets
TARGET = test_regression_qt
TEMPLATE = app
CONFIG += c++14

SOURCES += \
	main.cpp\
	mainwindow.cpp \
    SimpleCurve.cpp \
    ../Activation.cpp \
    ../Layer.cpp \
    ../Net.cpp \
    ../NetTrainMomentum.cpp \
    ../NetUtil.cpp \
    ../MatrixUtil.cpp \
    ../MNISTReader.cpp \
    ../ActivationLayer.cpp \
    ../DNNEngineTestDnn.cpp \
    ../DNNEngineTinyDnn.cpp \
    ../DNNEngine.cpp

HEADERS  += \
	mainwindow.h \
    SimpleCurve.h \
    ../Activation.h \
    ../Layer.h \
    ../Matrix.h \
    ../MatrixUtil.h \
    ../Net.h \
    ../NetTrainMomentum.h \	
    ../NetUtil.h \
    ../MNISTReader.h \
    ../ActivationLayer.h \
    ../DNNEngine.h \
    ../DNNEngineTestDnn.h \
    ../DNNEngineTinyDnn.h

FORMS    += mainwindow.ui

INCLUDEPATH += ..\

#tiny-dnn
INCLUDEPATH += $$(TINY_DNN_PATH)
