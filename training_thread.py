import traceback
from PyQt5.QtCore import QThread, pyqtSignal

class TrainingThread(QThread):
    training_finished = pyqtSignal(object)
    training_failed = pyqtSignal(str)

    def __init__(self, model, params, parent=None):
        super().__init__(parent)
        self.model = model
        self.params = params

    def run(self):
        try:
            # Start training. Logs will be printed to the console.
            results = self.model.train(**self.params)
            self.training_finished.emit(results)

        except Exception:
            exc_str = traceback.format_exc()
            self.training_failed.emit(exc_str)
