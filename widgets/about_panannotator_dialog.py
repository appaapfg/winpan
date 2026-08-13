from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

import panopticon.config


class AboutPanAnnotatorDialog(QDialog):
    def __init__(self, tr_function, parent=None):
        super().__init__(parent)
        self.tr = tr_function
        self.setWindowTitle(self.tr("about_panannotator_title"))

        layout = QVBoxLayout(self)

        logo_label = QLabel()
        pixmap = QPixmap("images/pan_annotator_logo.png")
        logo_label.setPixmap(pixmap.scaledToWidth(200, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        title_label = QLabel("Pan Annotator")
        title_label.setAlignment(Qt.AlignCenter)
        font = title_label.font()
        font.setPointSize(20)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        version_label = QLabel(self.tr("about_panopticon_version", version=panopticon.config.VERSION))
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        description_label = QLabel(self.tr("about_panannotator_description"))
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        author_label = QLabel(self.tr("about_panopticon_author"))
        author_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(author_label)

        license_label = QLabel(self.tr("about_panopticon_license"))
        license_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(license_label)

        ok_button = QPushButton(self.tr("ok_button"))
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.setMinimumWidth(350)
