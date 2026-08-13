import json

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class RemapIDsDialog(QDialog):
    """Dialog to build category ID remapping interactively.

    User selects an existing category, edits target id/name/supercategory, and adds it to mapping.
    Result is returned as JSON string matching previous expected structure.
    """

    def __init__(self, tr_function, parent=None, categories=None):
        super().__init__(parent)
        self.tr = tr_function
        self.setWindowTitle(self.tr("remap_ids_dialog_title"))
        self._mapping = {}  # old_id -> {id,name,supercategory}
        self._categories = categories or (parent.coco_data.get("categories", []) if parent and hasattr(parent, "coco_data") else [])
        self.layout = QVBoxLayout(self)

        # Add a detailed description label at the top
        description_label = QLabel(self.tr("remap_ids_dialog_description"))
        description_label.setWordWrap(True)
        self.layout.addWidget(description_label)

        self.layout.addWidget(QLabel(self.tr("existing_categories_label")))
        self.categories_list = QListWidget()
        for cat in self._categories:
            item = QListWidgetItem(f"{cat.get('id')} - {cat.get('name')}")
            item.setData(0x0100, cat)  # Qt.UserRole
            self.categories_list.addItem(item)
        self.layout.addWidget(self.categories_list)

        # Edit area for selected category's new mapping
        self.layout.addWidget(QLabel(self.tr("new_mapping_fields_label")))
        form_row1 = QHBoxLayout()
        form_row2 = QHBoxLayout()

        form_row1.addWidget(QLabel(self.tr("new_id_label")))
        self.new_id_spin = QSpinBox()
        self.new_id_spin.setRange(0, 10_000_000)
        form_row1.addWidget(self.new_id_spin)

        form_row2.addWidget(QLabel(self.tr("new_name_label")))
        self.new_name_edit = QLineEdit()
        form_row2.addWidget(self.new_name_edit)

        form_row2.addWidget(QLabel(self.tr("new_supercategory_label")))
        self.new_super_edit = QLineEdit()
        form_row2.addWidget(self.new_super_edit)

        self.layout.addLayout(form_row1)
        self.layout.addLayout(form_row2)

        self.add_mapping_btn = QPushButton(self.tr("add_mapping_button"))
        self.layout.addWidget(self.add_mapping_btn)

        self.layout.addWidget(QLabel(self.tr("current_mappings_label")))
        self.mappings_list = QListWidget()
        self.layout.addWidget(self.mappings_list)

        self.allow_collisions_cb = QCheckBox(self.tr("allow_collisions_label"))
        self.layout.addWidget(self.allow_collisions_cb)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(self.tr("ok_button"))
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText(self.tr("cancel_button"))
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        # Signals
        self.categories_list.currentItemChanged.connect(self._on_category_selected)
        self.add_mapping_btn.clicked.connect(self._add_or_update_mapping)
        self.mappings_list.itemDoubleClicked.connect(self._remove_mapping_item)

    def _on_category_selected(self, current, previous):
        if not current:
            return
        cat = current.data(0x0100)
        if not cat:
            return
        old_id = cat.get("id")
        # Prefill with existing mapping if present; else existing category values
        mapped = self._mapping.get(old_id, {"id": cat.get("id"), "name": cat.get("name"), "supercategory": cat.get("supercategory", "")})
        self.new_id_spin.setValue(int(mapped.get("id", 0)))
        self.new_name_edit.setText(str(mapped.get("name", "")))
        self.new_super_edit.setText(str(mapped.get("supercategory", "")))

    def _add_or_update_mapping(self):
        item = self.categories_list.currentItem()
        if not item:
            return
        cat = item.data(0x0100)
        old_id = cat.get("id")
        new_id = int(self.new_id_spin.value())
        new_name = self.new_name_edit.text().strip() or cat.get("name")
        new_super = self.new_super_edit.text().strip() or cat.get("supercategory", "")
        self._mapping[old_id] = {"id": new_id, "name": new_name, "supercategory": new_super}
        self._refresh_mappings_list()

    def _refresh_mappings_list(self):
        self.mappings_list.clear()
        for old_id, new_cat in sorted(self._mapping.items(), key=lambda x: x[0]):
            disp = f"{old_id} -> {new_cat['id']} ({new_cat['name']} / {new_cat['supercategory']})"
            li = QListWidgetItem(disp)
            li.setData(0x0100, old_id)
            self.mappings_list.addItem(li)

    def _remove_mapping_item(self, item):
        old_id = item.data(0x0100)
        if old_id in self._mapping:
            self._mapping.pop(old_id)
            self._refresh_mappings_list()

    def get_values(self):
        # Build JSON string matching expected structure
        data = {str(k): v for k, v in self._mapping.items()}
        mapping_json = json.dumps(data)
        return {
            "category_mapping": mapping_json,
            "allow_collisions": self.allow_collisions_cb.isChecked(),
        }
