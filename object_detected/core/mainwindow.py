# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import cv2
from .calcdata import calcimage
from .Ui_mainwindow import Ui_MainWindow
import os.path
import time
from typing import List, Tuple
import numpy as np

imgtranslate = []


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    file_path = "D:\\kmol\\testcode\\AItest\\objectdect\\gcodetmp\\test.txt"
    stoppath = "D:\\kmol\\testcode\\AItest\\objectdect\\gcodetmp\\stop.txt"
    gcodepath = "D:\\kmol\\testcode\\AItest\\objectdect\\gcodetmp\\gcode.txt"
    innermtx  = np.array(([716.83033741,   0., 306.26006964],
                          [0., 720.92700541, 249.84924506],
                          [0.,   0., 1.]))

    dist = np.array(([5.78261896e-02, -6.92817907e-01,  1.07652301e-03, -7.75407739e-04,  1.64367821e+00]))

    count = 0

    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.screwlsit = []
        self.allbb = []
    
    @pyqtSlot()
    def on_test_btn_clicked(self):
        
        dlg = QProgressDialog("progress bar", "cancel", 0, 3, self)
        dlg.show()
        QCoreApplication.processEvents()
        cv_image = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
        bytedata = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_image.shape
        bytesPerLine = 3 * width
        qimage = QImage(bytedata.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.picture1.setPixmap(QPixmap.fromImage(qimage).scaled(320, 240))
        dlg.setValue(1)
        QCoreApplication.processEvents()
        ccdata, data = calcimage(cv_image)
        bytedata = cv2.cvtColor(ccdata, cv2.COLOR_BGR2RGB)
        qproimg = QImage(bytedata.data, width, height, bytesPerLine, QImage.Format_RGB888)
        qproimg.save("test1.jpg")
        self.picture2.setPixmap(QPixmap.fromImage(qproimg).scaled(320, 240))
        dlg.setValue(2)
        im_width, im_height = cv_image.shape[:2]
        self.settable(data, im_width, im_height)
        dlg.setValue(3)
        with open("test_point.txt", 'w') as f:
            for i in self.allbb:
                f.write(f"{i[0]} {i[1]}\n")
    
    def opencv_initdis(self,files, mtx, dist):
        img = cv2.imread(files)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        mapx,mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "data2.jpg", "Text files (*.jpg)")
        if filename:
            cv2.imwrite(filename, dst)
    
    def settable(self, data, img_heigh, img_width):
        self.screwtable.clearContents()
        self.nuttable.clearContents()
        self.nuttable.setColumnCount(4)
        self.screwtable.setColumnCount(4)
        self.nuttable.setRowCount(61)
        self.screwtable.setRowCount(61)
        nut = 0
        screw = 0
        self.screwlsit.clear()
        self.allbb.clear()
        for box, color in data:
            ymin, xmin, ymax, xmax = box
            if color == "Chartreuse":
                for i in range(0, 4):
                    if i % 2 == 1:
                        tmp = box[i]*img_width
                    else :
                        tmp = box[i]*img_heigh
                    self.nuttable.setItem(nut, i, QTableWidgetItem(str(tmp)))
                nut += 1
            if color == "Aqua":
                for i in range(0, 4):
                    if i %2 == 1:
                        tmp = box[i]*img_width
                    else :
                        tmp = box[i]*img_heigh
                    self.screwtable.setItem(screw, i, QTableWidgetItem(str(tmp)))
                self.screwlsit.append((((xmin+xmax)*img_width)/2, ((ymin+ymax)*img_heigh)/2))
                screw+=1
            self.allbb.append((((xmin+xmax)*img_width)/2, ((ymin+ymax)*img_heigh)/2))
            #self.tableWidget
    
    @pyqtSlot()
    def on_chachimg_btn_clicked(self):
        cap = cv2.VideoCapture(self.cam_count.value())
        ret, frame = cap.read()
        cv2.imwrite("test.jpg", frame)
        bytedata = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qimage = QImage(bytedata.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.picture1.setPixmap(QPixmap.fromImage(qimage).scaled(320, 240))
        cap.release()
    
    @pyqtSlot(bool)
    def on_autorad_btn_toggled(self, check):
        if not check:
            return

        dlg = QProgressDialog("progress bar", "cancel", 0, 1, self)
        dlg.show()
        try:
            print("i'm start")
            while not os.path.isfile(self.file_path):
                if os.path.isfile(self.stoppath):
                    time.sleep(1)
                    os.remove(self.stoppath)
                    raise KeyboardInterrupt
                QCoreApplication.processEvents()
            else:
                time.sleep(1)
                os.remove(self.file_path)
                self.count += 1
                dlg.setValue(1)
                self.processimg()
                self.on_autorad_btn_toggled(True)
        except KeyboardInterrupt:
            dlg.setValue(1)
    
    @pyqtSlot()
    def on_genert_btn_clicked(self):
        # test = [(100, 200, 500), (90, 190, 500)]
        self.generGcode(self.catchpoint)
    
    def processimg(self):
        """
        cap = cv2.VideoCapture(self.cam_count.value())
        ret, frame = cap.read()
        cv2.imwrite("test.jpg", frame)
        cap.release()
        """
        
        dlg = QProgressDialog("progress bar", "cancel", 0, 3, self)
        dlg.show()
        QCoreApplication.processEvents()
        cv_image = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
        bytedata = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_image.shape
        bytesPerLine = 3 * width
        qimage = QImage(bytedata.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.picture1.setPixmap(QPixmap.fromImage(qimage).scaled(320, 240))
        dlg.setValue(1)
        QCoreApplication.processEvents()
        ccdata, data = calcimage(cv_image)
        bytedata = cv2.cvtColor(ccdata, cv2.COLOR_BGR2RGB)
        qproimg = QImage(bytedata.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.picture2.setPixmap(QPixmap.fromImage(qproimg).scaled(320, 240))
        QCoreApplication.processEvents()
        dlg.setValue(2)
        im_width, im_height = cv_image.shape[:2]
        self.settable(data, im_width, im_height)
        dlg.setValue(3)

    def generGcode(self, poistions: List[Tuple[float, float, float]]):
        
        # 位置 ps = start point 
        # droppoint = 放下點
        # raiseup = 提刀點
        
        ps = (407.2215, -272.5603, 331.3164)
        raiseup = (407.2215, -272.5603, 331.3164)
        droppoint = (407.2215, 272.56, 331)
        
        # 姿態 ABC
        left_edge = (50, 180, 10)
        right_edge = (50, 180, 10)
        top_edge = (50, 180, 10)
        bottom_edge = (50, 180, 10)
        normal_edge = (0, 180, 0)
        
        # 角落姿態
        upleftcorner = (0, 0, 0)
        lowerleftcorner = (0, 0, 0)
        uprightcorner = (0, 0, 0)
        lowerrightcorner = (0, 0, 0)
        
        # chacth range
        xMax = 568.0
        xMin = 364.0
        yMax = -188.0
        yMin = -367.0
        
        def r(m: int, x: float, y: float) -> float:
            if x < xMin:
                if x < xMin and y < yMin:
                    return upleftcorner[m]
                elif x < xMin and y > yMax:
                    return lowerleftcorner
                else:
                    return left_edge[m]
            elif x > xMax:
                if x > xMax and y < yMin:
                    return uprightcorner[m]
                elif x > xMax and y > yMin:
                    return lowerrightcorner[m]
                else:
                    return right_edge[m]
            elif y > yMin:
                return bottom_edge[m]
            elif y < yMax:
                return top_edge[m]
            return normal_edge[m]
        
        text = ''.join(
            f"P, X {ps[0]:.03f}, Y {ps[1]:.03f}, Z {ps[2]:.03f}, RA 0, RB 180, RC 0, V 5;\n"
            f"P, X {x:.03f}, Y {y:.03f}, Z {z:.03f}, RA {r(0,x,y)}, RB {r(1,x,y)}, RC {r(2,x,y)}, V 5;\n"
            f"OA 1;\n"
            f"P, X {raiseup[0]:.03f}, Y {raiseup[1]:.03f}, Z {raiseup[2]:.03f}, RA {r(0,x,y)}, RB {r(1,x,y)}, RC {r(2,x,y)}, V 5;\n"
            f"P, X {droppoint[0]:.03f}, Y {droppoint[1]:.03f}, Z {droppoint[2]:.03f}, RA 0, RB 180, RC 0, V 5;\n"
            f"OA 0;\n"
            for x, y, z in poistions
        )
        with open(self.gcodepath, 'w') as f:
            f.write(text)

    @pyqtSlot(bool)
    def on_custom_btn_toggled(self, checked):
        """
        Slot documentation goes here.
        
        @param checked DESCRIPTION
        @type bool
        """
        # TODO: not implemented yet
        pass
    
    @pyqtSlot()
    def on_testbtn_clicked(self):
        filename = "data.jpg"
        if filename:
            self.opencv_initdis(filename, self.innermtx, self.dist)
    
    @pyqtSlot()
    def on_calcpix2glo_btn_clicked(self):
        print(self.screwlsit)
        self.catchpoint = []
        test = np.array((  [[0.015984, -0.4931, 0],
                            [-0.4935, -0.025, 0],
                            [596.56166, -83.9282, 1]]))
        for i in self.screwlsit:            
            endpoint = np.dot([i[0], i[1], 1], test)
            print(endpoint)
            self.catchpoint.append(endpoint)
        # x translate 簡學長的方法
        #endpoint =  [pixelx, pixely, 1]*self.xtranslate
