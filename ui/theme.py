from __future__ import annotations

APP_STYLESHEET = """
QWidget {
    background: #f7efe4;
    color: #1e1b18;
    font-family: "Segoe UI Variable", "Microsoft YaHei UI", "PingFang SC";
    font-size: 14px;
}
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #f6e9db, stop:0.45 #f3dac6, stop:1 #dbe6db);
}
QFrame#Sidebar {
    background: rgba(28, 24, 21, 0.88);
    border-radius: 26px;
}
QFrame#Card {
    background: rgba(255, 249, 241, 0.86);
    border: 1px solid rgba(120, 90, 60, 0.15);
    border-radius: 24px;
}
QPushButton {
    border: none;
    border-radius: 16px;
    padding: 12px 18px;
    background: #e9d4ba;
    color: #2a2017;
    font-weight: 600;
}
QPushButton:hover {
    background: #e3c5a5;
}
QPushButton:pressed {
    background: #d4af86;
}
QPushButton#PrimaryButton {
    background: #1f6f5f;
    color: #f9f6f0;
}
QPushButton#PrimaryButton:hover {
    background: #2a7c6c;
}
QPushButton#GhostButton {
    background: rgba(255, 255, 255, 0.08);
    color: #f8efe5;
    text-align: left;
    padding: 14px 18px;
}
QPushButton#GhostButton:checked,
QPushButton#GhostButton:hover {
    background: rgba(255, 255, 255, 0.18);
}
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: rgba(255, 252, 246, 0.95);
    border: 1px solid rgba(70, 50, 30, 0.15);
    border-radius: 14px;
    padding: 10px 12px;
}
QComboBox::drop-down {
    border: none;
}
QListWidget {
    background: rgba(255, 252, 246, 0.94);
    border-radius: 18px;
    border: 1px solid rgba(70, 50, 30, 0.12);
    padding: 8px;
}
QListWidget::item {
    padding: 12px;
    margin: 4px;
    border-radius: 12px;
}
QListWidget::item:selected {
    background: #efe0ca;
    color: #241a11;
}
QProgressBar {
    background: rgba(255, 252, 246, 0.95);
    border: 1px solid rgba(70, 50, 30, 0.12);
    border-radius: 12px;
    text-align: center;
    min-height: 18px;
    color: #2a2017;
}
QProgressBar::chunk {
    border-radius: 10px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6f5f, stop:1 #e2a458);
}
QGroupBox {
    font-size: 15px;
    font-weight: 700;
    border: 1px solid rgba(70, 50, 30, 0.12);
    border-radius: 18px;
    margin-top: 12px;
    padding-top: 14px;
    background: rgba(255, 250, 244, 0.82);
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
}
QLabel#HeroTitle {
    font-size: 32px;
    font-weight: 800;
    color: #1d1712;
}
QLabel#HeroSubTitle {
    font-size: 15px;
    color: #67584b;
}
QLabel#SectionTitle {
    font-size: 18px;
    font-weight: 800;
}
QLabel#MetricLabel {
    color: #6a5a4d;
    font-size: 13px;
}
QLabel#MetricValue {
    font-size: 26px;
    font-weight: 800;
    color: #1f6f5f;
}
QScrollArea {
    border: none;
    background: transparent;
}
"""
