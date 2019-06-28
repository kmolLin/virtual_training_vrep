# Start detected
# Including Python module: PyQt5, pygraphy, pyopengl
# Copyright (C) 2017 you shang [pyquino@gmail.com]
from core.mainwindow import MainWindow
from PyQt5.QtWidgets import QApplication
from sys import argv, exit

if __name__=='__main__':
    QApplication.setStyle('fusion')
    app = QApplication(argv)
    run = MainWindow()
    run.show()
    exit(app.exec_())
