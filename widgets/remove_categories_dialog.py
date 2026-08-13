from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class RemoveCategoriesDialog(QDialog):
    """Dialog to select one or more categories to remove."""

    def __init__(self, parent, categories=None):
        super().__init__(parent)
        self.tr = self.parent().tr
        self.setWindowTitle(self.tr("remove_categories_dialog_title"))

        layout = QVBoxLayout(self)

        # Add a detailed description label at the top
        description_label = QLabel(self.tr("remove_categories_dialog_description"))
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        layout.addWidget(QLabel(self.tr("remove_categories_dialog_label")))

        self.category_list = QListWidget(self)
        self.category_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        if categories:
            for cat in categories:
                text = f"{cat.get('id')} - {cat.get('name')}"
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, cat.get("id"))
                item.setCheckState(Qt.CheckState.Unchecked)
                self.category_list.addItem(item)
        layout.addWidget(self.category_list)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)

        self.select_all_button = QPushButton(self.tr("select_all_button"))
        self.select_no_button = QPushButton(self.tr("select_no_button"))

        self.button_box.addButton(self.select_all_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.button_box.addButton(self.select_no_button, QDialogButtonBox.ButtonRole.ActionRole)

        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.tr("ok_button"))
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self.tr("cancel_button"))
        layout.addWidget(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.select_all_button.clicked.connect(self.select_all_categories)
        self.select_no_button.clicked.connect(self.select_no_categories)

        self.setLayout(layout)

    def select_all_categories(self):
        for i in range(self.category_list.count()):
            item = self.category_list.item(i)
            item.setCheckState(Qt.CheckState.Checked)

    def select_no_categories(self):
        for i in range(self.category_list.count()):
            item = self.category_list.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)

    def get_selection(self):
        selected_ids = []
        for i in range(self.category_list.count()):
            item = self.category_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_ids.append(item.data(Qt.ItemDataRole.UserRole))
        return {
            "category_ids": selected_ids,
            # "remove_annotations": True,
        }
