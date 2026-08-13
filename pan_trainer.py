import os
import sys

from PySide6.QtGui import QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
)

from panopticon.trainer_tabs.export_tab import ExportTab
from panopticon.trainer_tabs.prune_tab import PruneTab
from panopticon.trainer_tabs.train_tab import TrainTab
from panopticon.trainer_tabs.validate_tab import ValidateTab
from panopticon.widgets.about_pantrainer_dialog import AboutPanTrainerDialog
from panopticon.translations_data import TRANSLATIONS
from panopticon.config import VERSION


class PanTrainer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TRANSLATIONS["en"]["pan_trainer_title"].format(version=VERSION))
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "..", "images", "pan_trainer_logo_transparent.png")))
        self.setGeometry(100, 100, 800, 600)

        # Supertab
        supertab = QTabWidget()
        supertab.setMinimumSize(200, 200)

        # Train supertab
        self.train_tab = TrainTab(self)
        supertab.addTab(self.train_tab, "Train")

        # Prune supertab (empty for now)
        self.validate_tab = ValidateTab(self)
        supertab.addTab(self.validate_tab, "Validate")

        # Prune supertab
        self.prune_tab = PruneTab(self)
        supertab.addTab(self.prune_tab, "Prune")

        # Export supertab
        self.export_tab = ExportTab(self)
        supertab.addTab(self.export_tab, "Export")

        self.setCentralWidget(supertab)

        self.create_menu_bar()

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        load_action = edit_menu.addAction("Load Train Config")
        load_action.triggered.connect(self.train_tab.load_train_config)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        save_action = edit_menu.addAction("Save Train Config")
        save_action.triggered.connect(self.train_tab.save_train_config)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        load_export_action = edit_menu.addAction("Load Export Config")
        load_export_action.triggered.connect(self.export_tab.load_export_config)
        save_export_action = edit_menu.addAction("Save Export Config")
        save_export_action.triggered.connect(self.export_tab.save_export_config)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

    def show_about(self):
        dialog = AboutPanTrainerDialog(self)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("../images/pan_trainer_logo_transparent.png"))

    window = PanTrainer()
    window.show()
    sys.exit(app.exec())
