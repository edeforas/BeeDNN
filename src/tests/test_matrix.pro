TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt


#eigen use (optional)
DEFINES+= "USE_EIGEN_NO"
INCLUDEPATH+=$$(EIGEN_PATH)


SOURCES += test_matrix.cpp \
    ../Matrix.cpp

INCLUDEPATH+=..

HEADERS += \
    ../Matrix.h
