#-------------------------------------------------
#
# Project created by QtCreator 2014-02-26T05:30:38
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = CannyEdgeDetector
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp


INCLUDEPATH += D://opencv//sources//opencv_bin//install//include

LIBS += D://opencv//sources//opencv_bin//bin//*.dll
