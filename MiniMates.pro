QT += qml quick sql quickcontrols2 websockets multimedia

CONFIG += qmltypes
QML_IMPORT_NAME = AAAAA
QML_IMPORT_MAJOR_VERSION = 1
INCLUDEPATH += src

SOURCES += main.cpp \
    src/communicate.cpp \
    src/humanassets.cpp \
    src/llmmodel.cpp \
    src/player.cpp \
    src/qmlbridge.cpp \
    src/sqlconversationmodel.cpp

INSTALLS += target


HEADERS += \
    src/communicate.h \
    src/humanassets.h \
    src/llmmodel.h \
    src/player.h \
    src/qmlbridge.h \
    src/sqlconversationmodel.h

RESOURCES += \
    resource.qrc

RC_ICONS = favicon.ico
