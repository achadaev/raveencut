"""RaveenCut — silence removal desktop app."""
import sys

from PyQt6.QtWidgets import QApplication

import core.i18n  # noqa: F401 — installs builtins._ and default language
from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("RaveenCut")
    w = MainWindow()
    w.show()
    sys.exit(getattr(app, "exec")())
