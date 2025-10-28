import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 애플리케이션 스타일을 설정하여 좀 더 현대적으로 보이게 합니다.
    app.setStyle('Fusion')
    
    main_window = MainWindow()
    main_window.show()
    
    sys.exit(app.exec_())
