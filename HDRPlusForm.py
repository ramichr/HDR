# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HDRPlusForm.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1202, 750)
        Form.setBaseSize(QtCore.QSize(0, 0))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("res/icon.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        Form.setStyleSheet("#Form {background: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 #2b639b, stop:1 #234075);}")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(20, 10, 20, 10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.NameBadge = QtWidgets.QLabel(Form)
        self.NameBadge.setText("")
        self.NameBadge.setPixmap(QtGui.QPixmap("res/name_badge.svg"))
        self.NameBadge.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.NameBadge.setObjectName("NameBadge")
        self.horizontalLayout.addWidget(self.NameBadge)
        self.HDRPlusLogo = QtWidgets.QLabel(Form)
        self.HDRPlusLogo.setText("")
        self.HDRPlusLogo.setPixmap(QtGui.QPixmap("res/icon.svg"))
        self.HDRPlusLogo.setScaledContents(False)
        self.HDRPlusLogo.setAlignment(QtCore.Qt.AlignCenter)
        self.HDRPlusLogo.setObjectName("HDRPlusLogo")
        self.horizontalLayout.addWidget(self.HDRPlusLogo)
        self.UniLogo = QtWidgets.QLabel(Form)
        self.UniLogo.setText("")
        self.UniLogo.setPixmap(QtGui.QPixmap("res/uni_logo.svg"))
        self.UniLogo.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing)
        self.UniLogo.setObjectName("UniLogo")
        self.horizontalLayout.addWidget(self.UniLogo)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.stackedWidget = QtWidgets.QStackedWidget(Form)
        self.stackedWidget.setObjectName("stackedWidget")
        self.StartPage = QtWidgets.QWidget()
        self.StartPage.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.StartPage.setObjectName("StartPage")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.StartPage)
        self.horizontalLayout_3.setContentsMargins(40, 32, 40, 50)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.text1 = QtWidgets.QTextBrowser(self.StartPage)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.text1.setFont(font)
        self.text1.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.text1.setSource(QtCore.QUrl("qrc:/:/text/p1.md"))
        self.text1.setObjectName("text1")
        self.horizontalLayout_2.addWidget(self.text1)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.text2 = QtWidgets.QTextBrowser(self.StartPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.text2.sizePolicy().hasHeightForWidth())
        self.text2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.text2.setFont(font)
        self.text2.setAutoFillBackground(False)
        self.text2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.text2.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.text2.setSource(QtCore.QUrl("qrc:/:/text/p2.md"))
        self.text2.setObjectName("text2")
        self.verticalLayout_3.addWidget(self.text2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.Execute = QtWidgets.QToolButton(self.StartPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.Execute.sizePolicy().hasHeightForWidth())
        self.Execute.setSizePolicy(sizePolicy)
        self.Execute.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Execute.setAutoFillBackground(False)
        self.Execute.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.Execute.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("res/execute.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap("res/execute_hov.svg"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon1.addPixmap(QtGui.QPixmap("res/execute_hov.svg"), QtGui.QIcon.Active, QtGui.QIcon.Off)
        self.Execute.setIcon(icon1)
        self.Execute.setIconSize(QtCore.QSize(150, 50))
        self.Execute.setCheckable(False)
        self.Execute.setAutoRaise(True)
        self.Execute.setArrowType(QtCore.Qt.NoArrow)
        self.Execute.setObjectName("Execute")
        self.horizontalLayout_4.addWidget(self.Execute)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        self.stackedWidget.addWidget(self.StartPage)
        self.MainPage = QtWidgets.QWidget()
        self.MainPage.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.MainPage.setObjectName("MainPage")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.MainPage)
        self.horizontalLayout_8.setContentsMargins(0, 40, 0, 20)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(50, -1, 0, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label = QtWidgets.QLabel(self.MainPage)
        self.label.setEnabled(True)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("res/input.svg"))
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label.setObjectName("label")
        self.verticalLayout_5.addWidget(self.label)
        self.uploadBut = QtWidgets.QToolButton(self.MainPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.uploadBut.sizePolicy().hasHeightForWidth())
        self.uploadBut.setSizePolicy(sizePolicy)
        self.uploadBut.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.uploadBut.setAutoFillBackground(False)
        self.uploadBut.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.uploadBut.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("res/upload.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon2.addPixmap(QtGui.QPixmap("res/upload_hov.svg"), QtGui.QIcon.Active, QtGui.QIcon.Off)
        self.uploadBut.setIcon(icon2)
        self.uploadBut.setIconSize(QtCore.QSize(120, 35))
        self.uploadBut.setCheckable(False)
        self.uploadBut.setAutoRaise(True)
        self.uploadBut.setArrowType(QtCore.Qt.NoArrow)
        self.uploadBut.setObjectName("uploadBut")
        self.verticalLayout_5.addWidget(self.uploadBut)
        self.horizontalLayout_7.addLayout(self.verticalLayout_5)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setContentsMargins(-1, 6, -1, 6)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.addItemImageStackWidget = QtWidgets.QStackedWidget(self.MainPage)
        self.addItemImageStackWidget.setObjectName("addItemImageStackWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.page)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.addItemImageBut = QtWidgets.QPushButton(self.page)
        self.addItemImageBut.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.addItemImageBut.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("res/add_item_image.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.addItemImageBut.setIcon(icon3)
        self.addItemImageBut.setIconSize(QtCore.QSize(256, 256))
        self.addItemImageBut.setFlat(True)
        self.addItemImageBut.setObjectName("addItemImageBut")
        self.horizontalLayout_10.addWidget(self.addItemImageBut)
        self.addItemImageStackWidget.addWidget(self.page)
        self.pag2 = QtWidgets.QWidget()
        self.pag2.setObjectName("pag2")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.pag2)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.frame_2 = QtWidgets.QFrame(self.pag2)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_2.setLineWidth(2)
        self.frame_2.setObjectName("frame_2")
        self.previewsGridLayout = QtWidgets.QGridLayout(self.frame_2)
        self.previewsGridLayout.setObjectName("previewsGridLayout")
        self.verticalLayout_8.addWidget(self.frame_2)
        self.addItemImageStackWidget.addWidget(self.pag2)
        self.horizontalLayout_9.addWidget(self.addItemImageStackWidget)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(70, -1, 70, -1)
        self.gridLayout.setHorizontalSpacing(21)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.brightLabel = QtWidgets.QLabel(self.MainPage)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.brightLabel.setFont(font)
        self.brightLabel.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.brightLabel.setObjectName("brightLabel")
        self.verticalLayout_11.addWidget(self.brightLabel)
        self.brightSlider = QtWidgets.QSlider(self.MainPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.brightSlider.sizePolicy().hasHeightForWidth())
        self.brightSlider.setSizePolicy(sizePolicy)
        self.brightSlider.setStyleSheet("QSlider::groove:horizontal {\n"
"background-color: #EBEBEB;\n"
"height: 20px;\n"
"border-radius: 10px;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background-color: #F75578;\n"
"border-radius: 1px;\n"
"border: solid 1px #D9E4DE;\n"
"margin: 9px 0px 9px 12px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"background-color: #D9E4DE;\n"
"border-radius: 1px;\n"
"margin: 9px 12px 9px 0px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"\n"
"background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"    stop:0 #EF3F54, stop:1 #F95C86);\n"
"height: 10px;\n"
"width: 10px;\n"
"margin: 3px 9px 3px 9px;\n"
"border-radius: 7px;\n"
"border: 2px solid #EBF1EE;\n"
"}\n"
"\n"
"\n"
"\n"
"")
        self.brightSlider.setProperty("value", 50)
        self.brightSlider.setOrientation(QtCore.Qt.Horizontal)
        self.brightSlider.setObjectName("brightSlider")
        self.verticalLayout_11.addWidget(self.brightSlider)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_11.addItem(spacerItem4)
        self.contLabel = QtWidgets.QLabel(self.MainPage)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.contLabel.setFont(font)
        self.contLabel.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.contLabel.setObjectName("contLabel")
        self.verticalLayout_11.addWidget(self.contLabel)
        self.contSlider = QtWidgets.QSlider(self.MainPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.contSlider.sizePolicy().hasHeightForWidth())
        self.contSlider.setSizePolicy(sizePolicy)
        self.contSlider.setMinimumSize(QtCore.QSize(0, 0))
        self.contSlider.setStyleSheet("QSlider::groove:horizontal {\n"
"background-color: #EBEBEB;\n"
"height: 20px;\n"
"border-radius: 10px;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background-color: #F75578;\n"
"border-radius: 1px;\n"
"border: solid 1px #D9E4DE;\n"
"margin: 9px 0px 9px 12px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"background-color: #D9E4DE;\n"
"border-radius: 1px;\n"
"margin: 9px 12px 9px 0px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"\n"
"background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"    stop:0 #EF3F54, stop:1 #F95C86);\n"
"height: 10px;\n"
"width: 10px;\n"
"margin: 3px 9px 3px 9px;\n"
"border-radius: 7px;\n"
"border: 2px solid #EBF1EE;\n"
"}\n"
"\n"
"\n"
"\n"
"")
        self.contSlider.setProperty("value", 50)
        self.contSlider.setOrientation(QtCore.Qt.Horizontal)
        self.contSlider.setObjectName("contSlider")
        self.verticalLayout_11.addWidget(self.contSlider)
        self.gridLayout.addLayout(self.verticalLayout_11, 0, 0, 1, 1)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.colLabel = QtWidgets.QLabel(self.MainPage)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.colLabel.setFont(font)
        self.colLabel.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.colLabel.setObjectName("colLabel")
        self.verticalLayout_10.addWidget(self.colLabel)
        self.colSlider = QtWidgets.QSlider(self.MainPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.colSlider.sizePolicy().hasHeightForWidth())
        self.colSlider.setSizePolicy(sizePolicy)
        self.colSlider.setMinimumSize(QtCore.QSize(0, 0))
        self.colSlider.setStyleSheet("QSlider::groove:horizontal {\n"
"background-color: #EBEBEB;\n"
"height: 20px;\n"
"border-radius: 10px;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background-color: #F75578;\n"
"border-radius: 1px;\n"
"border: solid 1px #D9E4DE;\n"
"margin: 9px 0px 9px 12px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"background-color: #D9E4DE;\n"
"border-radius: 1px;\n"
"margin: 9px 12px 9px 0px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"\n"
"background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"    stop:0 #EF3F54, stop:1 #F95C86);\n"
"height: 10px;\n"
"width: 10px;\n"
"margin: 3px 9px 3px 9px;\n"
"border-radius: 7px;\n"
"border: 2px solid #EBF1EE;\n"
"}\n"
"\n"
"\n"
"\n"
"")
        self.colSlider.setProperty("value", 50)
        self.colSlider.setOrientation(QtCore.Qt.Horizontal)
        self.colSlider.setInvertedAppearance(False)
        self.colSlider.setObjectName("colSlider")
        self.verticalLayout_10.addWidget(self.colSlider)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_10.addItem(spacerItem5)
        self.wbLabel = QtWidgets.QLabel(self.MainPage)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.wbLabel.setFont(font)
        self.wbLabel.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.wbLabel.setObjectName("wbLabel")
        self.verticalLayout_10.addWidget(self.wbLabel)
        self.wbSlider = QtWidgets.QSlider(self.MainPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.wbSlider.sizePolicy().hasHeightForWidth())
        self.wbSlider.setSizePolicy(sizePolicy)
        self.wbSlider.setStyleSheet("QSlider::groove:horizontal {\n"
"background-color: #EBEBEB;\n"
"height: 20px;\n"
"border-radius: 10px;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"    background-color: #F75578;\n"
"border-radius: 1px;\n"
"border: solid 1px #D9E4DE;\n"
"margin: 9px 0px 9px 12px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"background-color: #D9E4DE;\n"
"border-radius: 1px;\n"
"margin: 9px 12px 9px 0px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"\n"
"background: qlineargradient(x1:0, y1:0, x2:0, y2:1,\n"
"    stop:0 #EF3F54, stop:1 #F95C86);\n"
"height: 10px;\n"
"width: 10px;\n"
"margin: 3px 9px 3px 9px;\n"
"border-radius: 7px;\n"
"border: 2px solid #EBF1EE;\n"
"}\n"
"\n"
"\n"
"\n"
"")
        self.wbSlider.setProperty("value", 50)
        self.wbSlider.setOrientation(QtCore.Qt.Horizontal)
        self.wbSlider.setObjectName("wbSlider")
        self.verticalLayout_10.addWidget(self.wbSlider)
        self.gridLayout.addLayout(self.verticalLayout_10, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem6 = QtWidgets.QSpacerItem(499, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem6)
        self.horizontalLayout_8.addLayout(self.verticalLayout)
        self.frame = QtWidgets.QFrame(self.MainPage)
        self.frame.setStyleSheet("color: #4469b2;")
        self.frame.setFrameShape(QtWidgets.QFrame.VLine)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setLineWidth(5)
        self.frame.setMidLineWidth(0)
        self.frame.setObjectName("frame")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_7.addItem(spacerItem7)
        self.startBut = QtWidgets.QToolButton(self.frame)
        self.startBut.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("res/start.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.startBut.setIcon(icon4)
        self.startBut.setIconSize(QtCore.QSize(128, 42))
        self.startBut.setAutoRaise(False)
        self.startBut.setObjectName("startBut")
        self.verticalLayout_7.addWidget(self.startBut)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_7.addItem(spacerItem8)
        self.horizontalLayout_8.addWidget(self.frame)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setContentsMargins(50, -1, 30, -1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        spacerItem9 = QtWidgets.QSpacerItem(10, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_6.addItem(spacerItem9)
        self.label_2 = QtWidgets.QLabel(self.MainPage)
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("res/output.svg"))
        self.label_2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_6.addWidget(self.label_2)
        spacerItem10 = QtWidgets.QSpacerItem(10, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_6.addItem(spacerItem10)
        self.downloadBut = QtWidgets.QToolButton(self.MainPage)
        self.downloadBut.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.downloadBut.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("res/download.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon5.addPixmap(QtGui.QPixmap("res/download_hov.svg"), QtGui.QIcon.Active, QtGui.QIcon.Off)
        self.downloadBut.setIcon(icon5)
        self.downloadBut.setIconSize(QtCore.QSize(120, 35))
        self.downloadBut.setAutoRaise(True)
        self.downloadBut.setObjectName("downloadBut")
        self.verticalLayout_6.addWidget(self.downloadBut)
        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem11)
        self.horizontalLayout_6.addLayout(self.verticalLayout_6)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem12)
        self.verticalLayout_4.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.outputStack = QtWidgets.QStackedWidget(self.MainPage)
        self.outputStack.setObjectName("outputStack")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.page_2)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_8 = QtWidgets.QLabel(self.page_2)
        self.label_8.setMaximumSize(QtCore.QSize(400, 400))
        self.label_8.setSizeIncrement(QtCore.QSize(1, 1))
        self.label_8.setText("")
        self.label_8.setPixmap(QtGui.QPixmap("res/image_hdr.svg"))
        self.label_8.setScaledContents(True)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_12.addWidget(self.label_8)
        self.outputStack.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.outputLayout = QtWidgets.QHBoxLayout(self.page_3)
        self.outputLayout.setObjectName("outputLayout")
        self.outputStack.addWidget(self.page_3)
        self.horizontalLayout_11.addWidget(self.outputStack)
        self.verticalLayout_4.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_8.addLayout(self.verticalLayout_4)
        self.stackedWidget.addWidget(self.MainPage)
        self.verticalLayout_2.addWidget(self.stackedWidget)
        self.verticalLayout_2.setStretch(0, 1)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)

        self.retranslateUi(Form)
        self.stackedWidget.setCurrentIndex(0)
        self.addItemImageStackWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "HDRPlus"))
        self.brightLabel.setText(_translate("Form", "Brightness"))
        self.contLabel.setText(_translate("Form", "Contrast"))
        self.colLabel.setText(_translate("Form", "Color Intensity"))
        self.wbLabel.setText(_translate("Form", "White Balance"))
        self.startBut.setText(_translate("Form", "..."))
import rc_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())