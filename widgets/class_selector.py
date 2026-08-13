from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIntValidator
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from panopticon.translations_data import TRANSLATIONS


class CustomIDDialog(QDialog):
    def __init__(self, tr_function, parent=None, initial_id=""):
        super().__init__(parent)
        self.tr = tr_function
        self.setWindowTitle(self.tr("custom_object_id_dialog_title"))

        layout = QFormLayout(self)

        self.id_input = QLineEdit(self)
        self.id_input.setPlaceholderText(self.tr("custom_object_id_dialog_id_placeholder"))
        self.id_input.setText(str(initial_id))
        self.id_input.setValidator(QIntValidator(0, 2147483647, self))
        layout.addRow(self.tr("custom_object_id_dialog_id_label"), self.id_input)

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText(self.tr("custom_object_id_dialog_name_placeholder"))
        layout.addRow(self.tr("custom_object_id_dialog_name_label"), self.name_input)

        self.supercategory_input = QLineEdit(self)
        self.supercategory_input.setPlaceholderText(self.tr("custom_object_id_dialog_supercategory_placeholder"))
        layout.addRow(self.tr("custom_object_id_dialog_supercategory_label"), self.supercategory_input)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        layout.addRow(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.setLayout(layout)

    def get_inputs(self):
        try:
            obj_id = int(self.id_input.text())
        except ValueError:
            obj_id = -1  # Indicate invalid ID
        obj_name = self.name_input.text().strip()
        obj_supercategory = self.supercategory_input.text().strip()
        return obj_id, obj_name, obj_supercategory


class ClassSelector(QWidget):
    # Emitted when a new custom category is added through the dialog.
    # Payload: {'id': int, 'name': str, 'supercategory': str}
    custom_category_added = Signal(dict)

    def __init__(self, categories, tr_function, current_language, parent=None):
        """
        Initializes the ClassSelector widget.

        Args:
            categories (list): A list of dictionaries, where each dictionary represents a category
                               and contains 'id', 'name', and 'supercategory'.
            tr_function (callable): A translation function to translate text.
            current_language (str): The current language code (e.g., "en", "gr").
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.tr = tr_function
        self.current_language = current_language
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.unselect_button = QPushButton(self.tr("unselect_button"))
        layout.addWidget(self.unselect_button)
        self.unselect_button.clicked.connect(self._unselect_current)

        self.category_list = QListWidget()
        layout.addWidget(self.category_list)

        self.custom_id_button = QPushButton(self.tr("custom_object_id_button"))
        layout.addWidget(self.custom_id_button)
        self.custom_id_button.clicked.connect(self._prompt_for_custom_id)

        self.populate_categories(categories)

        self.current_category_id = None
        self.current_category_name = None
        self.current_category_supercategory = None
        self._custom_id_mode = False
        self.setLayout(layout)

    def _unselect_current(self):
        self.category_list.clearSelection()
        self.current_category_id = None
        self.current_category_name = None
        self.current_category_supercategory = None

    def tr(self, text, **kwargs):
        # Access TRANSLATIONS from the module scope
        if self.current_language in TRANSLATIONS:
            translation = TRANSLATIONS[self.current_language].get(text, text)
        else:
            translation = TRANSLATIONS["en"].get(text, text)

        # Ensure translation is always a string
        if translation is None:
            translation = text

        if kwargs:
            try:
                return str(translation).format(**kwargs)
            except KeyError as e:
                print(f"Translation key error: {e} for text: '{text}'")
                return str(text)
        return str(translation)

    def populate_categories(self, categories):
        self.category_list.clear()

        # Group categories by supercategory
        supercategory_groups = {}
        other_categories = []

        for cat in categories:
            supercategory = cat.get("supercategory", "").strip()
            if not supercategory:
                other_categories.append(cat)
            else:
                if supercategory not in supercategory_groups:
                    supercategory_groups[supercategory] = []
                supercategory_groups[supercategory].append(cat)

        # Sort supercategories alphabetically
        sorted_supercategories = sorted(supercategory_groups.keys())

        # Add supercategories and their categories
        for supercategory in sorted_supercategories:
            # Add supercategory header (bold, non-selectable)
            supercategory_item = QListWidgetItem(supercategory)
            font = QFont()
            font.setBold(True)
            supercategory_item.setFont(font)
            supercategory_item.setFlags(supercategory_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.category_list.addItem(supercategory_item)

            # Add categories under this supercategory (sorted alphabetically)
            sorted_cats = sorted(supercategory_groups[supercategory], key=lambda x: x.get("name", ""))
            for cat in sorted_cats:
                display_text = f"{cat['id']} - {cat['name']}"
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, cat["id"])
                item.setData(Qt.ItemDataRole.UserRole + 1, cat["name"])
                item.setData(Qt.ItemDataRole.UserRole + 2, cat.get("supercategory", ""))
                self.category_list.addItem(item)

        # Add "other" section for categories without supercategory
        if other_categories:
            # Add "other" header (bold, non-selectable)
            other_item = QListWidgetItem("other")
            font = QFont()
            font.setBold(True)
            other_item.setFont(font)
            other_item.setFlags(other_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.category_list.addItem(other_item)

            # Add categories under "other" (sorted alphabetically)
            sorted_other = sorted(other_categories, key=lambda x: x.get("name", ""))
            for cat in sorted_other:
                display_text = f" {cat['id']} - {cat['name']}"
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, cat["id"])
                item.setData(Qt.ItemDataRole.UserRole + 1, cat["name"])
                item.setData(Qt.ItemDataRole.UserRole + 2, cat.get("supercategory", ""))
                self.category_list.addItem(item)

    def _prompt_for_custom_id(self):
        """
        Prompts the user for a custom object ID, name, and supercategory using a custom dialog.
        If valid inputs are entered, it sets the custom ID mode, disables the category list,
        updates the custom ID button text, and adds the custom ID, name, and supercategory to the list for display.
        """
        existing_ids = [self.category_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.category_list.count()) if self.category_list.item(i).data(Qt.ItemDataRole.UserRole) is not None]
        next_id = (max(existing_ids) + 1) if existing_ids else 0

        dialog = CustomIDDialog(self.tr, self, initial_id=str(next_id))
        if dialog.exec() == QDialog.DialogCode.Accepted:
            num, name, supercategory = dialog.get_inputs()
            if num >= 0 and name:  # Ensure ID is valid and name is not empty
                self.current_category_id = num
                self.current_category_name = name
                self.current_category_supercategory = supercategory
                self.category_list.clearSelection()
                self.custom_id_button.setText(self.tr("custom_object_id_button"))
                # Add to list for display if it's a new custom ID
                if not any(self.category_list.item(i).data(Qt.ItemDataRole.UserRole) == num for i in range(self.category_list.count())):
                    display_text = f"{num} - {name}"
                    item = QListWidgetItem(display_text)
                    item.setData(Qt.ItemDataRole.UserRole, num)
                    item.setData(Qt.ItemDataRole.UserRole + 1, name)
                    item.setData(Qt.ItemDataRole.UserRole + 2, supercategory)
                    self.category_list.addItem(item)
                    # Emit signal so parent (PanAnnotator) can update its COCO categories
                    self.custom_category_added.emit(
                        {
                            "id": num,
                            "name": name,
                            "supercategory": supercategory,
                        }
                    )
                    # Re-populate to ensure correct grouping by supercategory
                    self.populate_categories(self.get_coco_categories_list())
            else:
                QMessageBox.warning(self, self.tr("warning_title"), self.tr("invalid_custom_id_or_name_message"))

    def get_current_selection(self, supress_warning=False):
        selected_items = self.category_list.selectedItems()
        if not selected_items:
            if not supress_warning:
                QMessageBox.warning(self, self.tr("warning_title"), self.tr("select_category_message"))
            return None, None

        item = selected_items[0]
        cat_id = item.data(Qt.ItemDataRole.UserRole)
        name = item.data(Qt.ItemDataRole.UserRole + 1)

        return name, cat_id

    def get_coco_categories_list(self) -> list:
        """
        Returns the dictionary of categories used to populate the list.

        Returns:
            list: A list of categories, where each category is a dictionary
                  containing 'id', 'name', and 'supercategory'.
        """
        categories = []
        for i in range(self.category_list.count()):
            item = self.category_list.item(i)
            cat_id = item.data(Qt.ItemDataRole.UserRole)
            # Skip header/grouping items which don't have category data set
            if cat_id is None:
                continue
            name = item.data(Qt.ItemDataRole.UserRole + 1)
            supercategory = item.data(Qt.ItemDataRole.UserRole + 2)
            categories.append({"id": cat_id, "name": name, "supercategory": supercategory})
        return categories

    def retranslate_ui(self):
        """Refresh translatable UI text elements.
        Only updates widget labels; category entries remain as originally added.
        """
        # Update button texts using provided translation function
        self.unselect_button.setText(self.tr("unselect_button"))
        self.custom_id_button.setText(self.tr("custom_object_id_button"))
