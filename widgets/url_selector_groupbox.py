from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from panopticon.favorites_utils import (
    FavoritesButtonsWidget,
    load_and_apply_favorite_settings,
)


class URLSelectorGroupBox(QGroupBox):
    def __init__(self, tr, status_bar, ui_elements, parent=None):
        super().__init__(tr("url_selector_groupbox_title"), parent)
        self.tr = tr
        self.status_bar = status_bar
        self.ui_elements = ui_elements
        url_selector_layout = QVBoxLayout(self)

        self.favorites_buttons_widget = FavoritesButtonsWidget(tr)
        url_selector_layout.addWidget(self.favorites_buttons_widget)

        connection_layout = QHBoxLayout()
        url_label = QLabel("")
        ui_elements["url_label"] = url_label
        self.url_input = QLineEdit()
        ui_elements["url_input"] = self.url_input
        connection_layout.addWidget(url_label)
        connection_layout.addWidget(self.url_input)
        url_selector_layout.addLayout(connection_layout)

        load_and_apply_favorite_settings(
            self.favorites_buttons_widget.get_buttons(),
            self.url_input,
            status_bar,
            tr,
        )

        self.connect_button = QPushButton("")
        ui_elements["connect_button"] = self.connect_button
        url_selector_layout.addWidget(self.connect_button)

    def retranslate_ui(self, is_opened):
        self.setTitle(self.tr("url_selector_groupbox_title"))
        self.ui_elements["url_label"].setText(self.tr("url_label"))
        self.ui_elements["url_input"].setPlaceholderText(self.tr("url_placeholder"))
        self.ui_elements["connect_button"].setText(self.tr("disconnect_button") if is_opened else self.tr("connect_button"))
        load_and_apply_favorite_settings(
            self.favorites_buttons_widget.get_buttons(),
            self.url_input,
            self.status_bar,
            self.tr,
        )
