import os

import yaml
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from panopticon.config import SETTINGS_FILE

DEFAULT_NUM_FAVORITES = 6
NUM_FAVORITES = DEFAULT_NUM_FAVORITES

# Load dynamic number of favorites from settings if provided
try:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as _f:
            _settings_loaded = yaml.safe_load(_f) or {}
            if isinstance(_settings_loaded, dict):
                _nf = _settings_loaded.get("number_of_favorites")
                if isinstance(_nf, int) and _nf > 0:
                    NUM_FAVORITES = _nf
except Exception:
    # Silently ignore and fall back to default
    pass


def load_and_apply_favorite_settings(fav_buttons, url_input_field, status_bar, tr_func):
    """Loads settings and updates favorite button texts and click actions."""
    if not os.path.exists(SETTINGS_FILE):
        for i, btn in enumerate(fav_buttons):  # Iterates up to len(fav_buttons)
            btn.setText(tr_func("fav_button_default_text", number=i + 1))
            btn.setToolTip(tr_func("fav_button_tooltip_configure"))
            btn.setEnabled(False)
        return

    try:
        with open(SETTINGS_FILE) as f:
            settings = yaml.safe_load(f)
            if settings and "favorites" in settings:
                favs = settings.get("favorites", [])
                # Configure buttons that have settings
                for i in range(min(len(favs), len(fav_buttons), NUM_FAVORITES)):  # Ensure we don't exceed any bounds
                    btn = fav_buttons[i]
                    if favs[i] and isinstance(favs[i], dict):
                        name = favs[i].get("name", "")
                        url = favs[i].get("url", "")

                        can_connect = bool(url)
                        btn.setEnabled(can_connect)

                        display_text = tr_func("fav_button_default_text", number=i + 1)
                        if name:
                            display_text = name
                        elif can_connect:  # Display URL if no name but URL exists
                            display_text = url

                        btn.setText(display_text[:20])

                        tooltip_parts = []
                        if name:
                            tooltip_parts.append(tr_func("fav_button_tooltip_name", name=name))
                        if url:
                            tooltip_parts.append(tr_func("fav_button_tooltip_connect_to", url=url))

                        if not tooltip_parts:
                            btn.setToolTip(tr_func("fav_button_tooltip_empty"))
                        else:
                            btn.setToolTip("\n".join(tooltip_parts))

                        if can_connect:
                            try:
                                btn.clicked.disconnect()
                            except RuntimeError:
                                pass
                            btn.clicked.connect(
                                lambda checked=False, url_val=url, b=btn: (
                                    url_input_field.setText(url_val),
                                    status_bar.showMessage(
                                        tr_func(
                                            "status_loaded_favorite",
                                            fav_name=b.toolTip().splitlines()[0],
                                        )
                                    ),
                                )
                            )
                        else:
                            try:
                                btn.clicked.disconnect()
                            except RuntimeError:
                                pass
                    else:  # favs[i] is None or not a dict
                        fav_buttons[i].setText(tr_func("fav_button_default_text", number=i + 1))
                        fav_buttons[i].setToolTip(tr_func("fav_button_tooltip_empty"))
                        fav_buttons[i].setEnabled(False)
                        try:
                            fav_buttons[i].clicked.disconnect()
                        except RuntimeError:
                            pass

                # For any remaining buttons beyond what's in favs (up to NUM_FAVORITES or len(fav_buttons))
                # This loop ensures that if favs has 2 entries, but we have 6 buttons, buttons 3-6 are reset.
                start_index_for_reset = min(len(favs), len(fav_buttons), NUM_FAVORITES)
                for i in range(start_index_for_reset, len(fav_buttons)):
                    fav_buttons[i].setText(tr_func("fav_button_default_text", number=i + 1))
                    fav_buttons[i].setToolTip(tr_func("fav_button_tooltip_empty"))  # Or configure tooltip
                    fav_buttons[i].setEnabled(False)
                    try:
                        fav_buttons[i].clicked.disconnect()
                    except RuntimeError:
                        pass
            else:  # No "favorites" key in settings or settings is None/empty
                for i, btn in enumerate(fav_buttons):
                    btn.setText(tr_func("fav_button_default_text", number=i + 1))
                    btn.setToolTip(tr_func("fav_button_tooltip_configure"))
                    btn.setEnabled(False)
                    try:
                        btn.clicked.disconnect()
                    except RuntimeError:
                        pass
    except Exception as e:
        print(f"Error applying settings to buttons: {e}")
        status_bar.showMessage(tr_func("status_error_loading_favorites", e=e), 5000)


class FavoritesButtonsWidget(QWidget):
    def __init__(self, tr_func, parent=None):
        super().__init__(parent)
        self.tr_func = tr_func
        self.fav_buttons = []
        self.layout = QGridLayout(self)  # Set the layout for this widget

        self._setup_ui()

    def _setup_ui(self):
        button_grid_cols = 3
        for i in range(NUM_FAVORITES):
            btn = QPushButton(self.tr_func("fav_button_default_text", number=i + 1))
            self.fav_buttons.append(btn)
            row = i // button_grid_cols
            col = i % button_grid_cols
            self.layout.addWidget(btn, row, col)

    def get_buttons(self):
        return self.fav_buttons

    def get_layout(self):
        return self.layout


class FavoritesDialog(QDialog):
    def __init__(self, parent, tr_func):
        super().__init__(parent)
        self.tr = tr_func
        self.setWindowTitle(self.tr("favorites_dialog_title"))
        self.setMinimumWidth(600)  # Adjusted for more fields

        # Add a detailed description label at the top
        description_label = QLabel(self.tr("favorites_dialog_description"))
        description_label.setWordWrap(True)

        self.name_inputs = []
        self.url_inputs = []

        layout = QVBoxLayout(self)
        layout.addWidget(description_label)
        # Use QGridLayout for the main form area to achieve 2x3 layout for 6 favorites
        form_fields_layout = QGridLayout()

        # Define the grid structure for favorite entries
        # 2 rows, 3 columns of favorite entries
        dialog_grid_cols = 3

        for i in range(NUM_FAVORITES):
            name_edit = QLineEdit()
            name_edit.setPlaceholderText(self.tr("favorites_dialog_name_placeholder"))
            url_edit = QLineEdit()
            url_edit.setPlaceholderText(self.tr("favorites_dialog_url_placeholder"))

            self.name_inputs.append(name_edit)
            self.url_inputs.append(url_edit)

            # Create a QFormLayout for each favorite's name and URL
            mini_form_layout = QFormLayout()
            mini_form_layout.addRow(self.tr("favorites_dialog_fav_n_name", number=i + 1), name_edit)
            mini_form_layout.addRow(self.tr("favorites_dialog_fav_n_url", number=i + 1), url_edit)

            # Wrap the mini_form_layout in a QWidget to add it to the QGridLayout
            fav_widget = QWidget()
            fav_widget.setLayout(mini_form_layout)

            # Calculate grid position (row-major filling)
            grid_row = i // dialog_grid_cols
            grid_col = i % dialog_grid_cols

            form_fields_layout.addWidget(fav_widget, grid_row, grid_col)

        layout.addLayout(form_fields_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Save).setText(self.tr("save_button"))
        button_box.button(QDialogButtonBox.Cancel).setText(self.tr("cancel_button"))
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.load_settings()

    def load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE) as f:
                    settings = yaml.safe_load(f)
                    if settings and "favorites" in settings:
                        favs = settings.get("favorites", [])
                        for i in range(min(len(favs), NUM_FAVORITES)):
                            if favs[i]:  # Check if entry is not None
                                self.name_inputs[i].setText(favs[i].get("name", ""))
                                self.url_inputs[i].setText(favs[i].get("url", ""))
        except Exception as e:
            print(f"Error loading favorites settings: {e}")
            # Optionally, inform the user via a status message or dialog

    def save_settings(self):
        all_settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    loaded_content = yaml.safe_load(f)
                    if isinstance(loaded_content, dict):
                        all_settings = loaded_content
                    else:
                        all_settings = {}  # Default to empty if file content is not a dict
            except Exception as e:
                print(f"Error reading settings file for saving favorites: {e}")
                all_settings = {}  # Default to empty if error

        current_favorites = []
        for i in range(NUM_FAVORITES):
            name = self.name_inputs[i].text()
            url = self.url_inputs[i].text()
            current_favorites.append({"name": name, "url": url})

        all_settings["favorites"] = current_favorites

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(all_settings, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Error saving settings: {e}")
            # Optionally, inform the user

    def accept(self):
        self.save_settings()
        super().accept()
