# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\kmol\testcode\AItest\objectdect\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(913, 631)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.picture1 = QtWidgets.QLabel(self.centralWidget)
        self.picture1.setMinimumSize(QtCore.QSize(320, 240))
        self.picture1.setText("")
        self.picture1.setAlignment(QtCore.Qt.AlignCenter)
        self.picture1.setObjectName("picture1")
        self.horizontalLayout.addWidget(self.picture1)
        self.picture2 = QtWidgets.QLabel(self.centralWidget)
        self.picture2.setMinimumSize(QtCore.QSize(320, 240))
        self.picture2.setText("")
        self.picture2.setAlignment(QtCore.Qt.AlignCenter)
        self.picture2.setObjectName("picture2")
        self.horizontalLayout.addWidget(self.picture2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.calcpix2glo_btn = QtWidgets.QPushButton(self.centralWidget)
        self.calcpix2glo_btn.setObjectName("calcpix2glo_btn")
        self.gridLayout.addWidget(self.calcpix2glo_btn, 0, 1, 1, 1)
        self.genert_btn = QtWidgets.QPushButton(self.centralWidget)
        self.genert_btn.setObjectName("genert_btn")
        self.gridLayout.addWidget(self.genert_btn, 1, 1, 1, 1)
        self.test_btn_2 = QtWidgets.QPushButton(self.centralWidget)
        self.test_btn_2.setObjectName("test_btn_2")
        self.gridLayout.addWidget(self.test_btn_2, 1, 0, 1, 1)
        self.test_btn = QtWidgets.QPushButton(self.centralWidget)
        self.test_btn.setObjectName("test_btn")
        self.gridLayout.addWidget(self.test_btn, 0, 0, 1, 1)
        self.testbtn = QtWidgets.QPushButton(self.centralWidget)
        self.testbtn.setObjectName("testbtn")
        self.gridLayout.addWidget(self.testbtn, 0, 2, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.custom_btn = QtWidgets.QRadioButton(self.centralWidget)
        self.custom_btn.setChecked(True)
        self.custom_btn.setObjectName("custom_btn")
        self.verticalLayout_4.addWidget(self.custom_btn)
        self.autorad_btn = QtWidgets.QRadioButton(self.centralWidget)
        self.autorad_btn.setObjectName("autorad_btn")
        self.verticalLayout_4.addWidget(self.autorad_btn)
        self.pauserad_btn = QtWidgets.QRadioButton(self.centralWidget)
        self.pauserad_btn.setObjectName("pauserad_btn")
        self.verticalLayout_4.addWidget(self.pauserad_btn)
        self.gridLayout.addLayout(self.verticalLayout_4, 1, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_7 = QtWidgets.QLabel(self.centralWidget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_3.addWidget(self.label_7)
        self.cam_count = QtWidgets.QSpinBox(self.centralWidget)
        self.cam_count.setObjectName("cam_count")
        self.horizontalLayout_3.addWidget(self.cam_count)
        self.chachimg_btn = QtWidgets.QPushButton(self.centralWidget)
        self.chachimg_btn.setObjectName("chachimg_btn")
        self.horizontalLayout_3.addWidget(self.chachimg_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.screwtable = QtWidgets.QTableWidget(self.centralWidget)
        self.screwtable.setObjectName("screwtable")
        self.screwtable.setColumnCount(4)
        self.screwtable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.screwtable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.screwtable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.screwtable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.screwtable.setHorizontalHeaderItem(3, item)
        self.verticalLayout_2.addWidget(self.screwtable)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.nuttable = QtWidgets.QTableWidget(self.centralWidget)
        self.nuttable.setObjectName("nuttable")
        self.nuttable.setColumnCount(4)
        self.nuttable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.nuttable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.nuttable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.nuttable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.nuttable.setHorizontalHeaderItem(3, item)
        self.verticalLayout_3.addWidget(self.nuttable)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.calcpix2glo_btn.setText(_translate("MainWindow", "calc pix2glo"))
        self.genert_btn.setText(_translate("MainWindow", "Genert"))
        self.test_btn_2.setText(_translate("MainWindow", "test"))
        self.test_btn.setText(_translate("MainWindow", "paress"))
        self.testbtn.setText(_translate("MainWindow", "testbtn"))
        self.custom_btn.setText(_translate("MainWindow", "Custom"))
        self.autorad_btn.setText(_translate("MainWindow", "Auto"))
        self.pauserad_btn.setText(_translate("MainWindow", "pause"))
        self.label_7.setText(_translate("MainWindow", "Webcam"))
        self.chachimg_btn.setText(_translate("MainWindow", "Chatch photo"))
        self.label.setText(_translate("MainWindow", "Screw"))
        item = self.screwtable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "ymin"))
        item = self.screwtable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "xmin"))
        item = self.screwtable.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "ymax"))
        item = self.screwtable.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "xmax"))
        self.label_2.setText(_translate("MainWindow", "nut"))
        item = self.nuttable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "ymin"))
        item = self.nuttable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "xmin"))
        item = self.nuttable.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "ymax"))
        item = self.nuttable.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "xmax"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

