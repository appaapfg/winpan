from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SuggestionControlWidget(QWidget):
    def __init__(self, tr_func, accept_callback, previous_callback, reject_callback, finish_callback, parent=None):
        super().__init__(parent)
        self.tr = tr_func
        self.accept_callback = accept_callback
        self.previous_callback = previous_callback
        self.reject_callback = reject_callback
        self.finish_callback = finish_callback
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.suggestion_progress_label = QLabel("0/0")
        layout.addWidget(self.suggestion_progress_label)
        btn_layout = QHBoxLayout()
        self.accept_suggestion_btn = QPushButton(self.tr("accept_button"))
        self.previous_suggestion_btn = QPushButton(self.tr("previous_button"))
        self.previous_suggestion_btn.setShortcut(QKeySequence(QKeySequence.StandardKey.Back))
        self.reject_suggestion_btn = QPushButton(self.tr("next_button"))
        self.reject_suggestion_btn.setShortcut(QKeySequence(QKeySequence.StandardKey.Forward))
        self.finish_suggestion_btn = QPushButton(self.tr("finish_button"))
        btn_layout.addWidget(self.accept_suggestion_btn)
        btn_layout.addWidget(self.previous_suggestion_btn)
        btn_layout.addWidget(self.reject_suggestion_btn)
        btn_layout.addWidget(self.finish_suggestion_btn)
        layout.addLayout(btn_layout)
        self.hint_label = QLabel(self.tr("suggestion_mode_hint"))
        layout.addWidget(self.hint_label)
        self.accept_suggestion_btn.clicked.connect(self.accept_callback)
        self.previous_suggestion_btn.clicked.connect(self.previous_callback)
        self.reject_suggestion_btn.clicked.connect(self.reject_callback)
        self.finish_suggestion_btn.clicked.connect(self.finish_callback)

    def update_progress(self, current, total):
        self.suggestion_progress_label.setText(f"{current}/{total}")

    def retranslate_ui(self):
        self.accept_suggestion_btn.setText(self.tr("accept_button"))
        self.previous_suggestion_btn.setText(self.tr("previous_button"))
        self.reject_suggestion_btn.setText(self.tr("next_button"))
        self.finish_suggestion_btn.setText(self.tr("finish_button"))
        self.hint_label.setText(self.tr("suggestion_mode_hint"))
