from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPainter, QPalette, QPen, QPixmap


def _draw_magnifier_icon(mode: str, palette: QPalette | None = None) -> QIcon:
    size = 32
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    color = palette.color(QPalette.ColorRole.WindowText) if palette is not None else Qt.GlobalColor.black

    pen = QPen(color)
    pen.setWidth(2)
    painter.setPen(pen)

    painter.drawEllipse(6, 4, 18, 18)
    painter.drawLine(19, 19, 28, 28)

    center_x, center_y = 15, 13
    if mode == "plus":
        painter.drawLine(center_x, center_y - 4, center_x, center_y + 4)
        painter.drawLine(center_x - 4, center_y, center_x + 4, center_y)
    elif mode == "minus":
        painter.drawLine(center_x - 4, center_y, center_x + 4, center_y)
    elif mode == "fit":
        painter.drawRect(9, 7, 12, 12)
        painter.drawLine(9, 13, 5, 13)
        painter.drawLine(21, 13, 25, 13)
        painter.drawLine(15, 7, 15, 3)
        painter.drawLine(15, 19, 15, 23)

    painter.end()
    return QIcon(pixmap)


def get_zoom_icon(theme_name: str, mode: str, palette: QPalette | None = None) -> QIcon:
    icon = QIcon.fromTheme(theme_name)
    if icon.isNull():
        icon = _draw_magnifier_icon(mode, palette)
    return icon
