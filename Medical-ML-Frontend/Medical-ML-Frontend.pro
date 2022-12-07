QT       += core gui network charts

RC_ICONS = media/icon.ico

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    login.cpp \
    cryptoUtil.cpp \
    main.cpp \
    mainwindow.cpp \
    hashing/sha256.cpp

HEADERS += \
    login.h \
    cryptoUtil.h \
    mainwindow.h \
    hashing/sha256.h

FORMS += \
    login.ui \
    mainwindow.ui

CONFIG += lrelease
CONFIG += embed_translations

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    resources.qrc


# cryptopp
INCLUDEPATH += C:\cryptolibs\cryptopp-CRYPTOPP_8_1_0

contains(QT_ARCH, i386) {
    message("32-bit")
    TARGETBIT=32bit
} else {
    message("64-bit")
    TARGETBIT=64bit
}

CONFIG(debug, debug|release) {
    LIBS += -L$$PWD/cryptpp/static/msvc2019_$$TARGETBIT/Debug -lcryptlib
    message($$LIBS)
}

CONFIG(release, debug|release) {
    LIBS += -L$$PWD/cryptpp/static/msvc2019_$$TARGETBIT/Release -lcryptlib
    message($$LIBS)
}

DISTFILES += \
    media/checked.png \
    media/close.png \
    media/warning.png
