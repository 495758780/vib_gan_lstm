import sys

import ui_2
from PyQt5.QtWidgets import QApplication,QMainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QMainWindow()
    ui = ui_2.Ui_MainWindow()
    ui.setupUi(main)
    main.show()
    sys.exit(app.exec_())
