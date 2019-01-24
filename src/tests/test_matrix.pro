TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

DEFINES+= "USE_EIGEN_NO"

SOURCES += test_matrix.cpp

INCLUDEPATH+=..
INCLUDEPATH+=$$(EIGEN_PATH)

HEADERS += \
    ../Matrix.h

