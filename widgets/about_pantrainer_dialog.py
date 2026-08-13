from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

import panopticon.config


class AboutPanTrainerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About PanTrainer")

        layout = QVBoxLayout(self)

        logo_label = QLabel()
        pixmap = QPixmap("images/pan_trainer_logo.png")
        logo_label.setPixmap(pixmap.scaledToWidth(200, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        title_label = QLabel("PanTrainer")
        title_label.setAlignment(Qt.AlignCenter)
        font = title_label.font()
        font.setPointSize(20)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        version_label = QLabel(f"Version: {panopticon.config.VERSION}")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        description_label = QLabel("PanTrainer - A GUI for training Ultralytics object detection models.")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        author_label = QLabel("Authors: Private (Signal Corp) NONE (Lead & Main Developer), Private (Signal Corp) NONE (Developer), Captain (Signal Corp) NONE (Non-Technical Manager)")
        author_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(author_label)

        license_label = QLabel("License: Greek Army Classified")
        license_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(license_label)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.setMinimumWidth(350)
