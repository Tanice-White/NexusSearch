from __future__ import annotations

import sys
from typing import Any

from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.service import NexusRAGService
from ui.pages import LibraryPage, NavButton, SearchPage, SettingsPage
from ui.settings_store import DesktopSettings, SettingsStore
from ui.theme import APP_STYLESHEET
from ui.workers import StreamTask, TaskRunner


class NexusDesktopWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.store = SettingsStore()
        self.settings = self.store.load()
        self.thread_pool = QThreadPool.globalInstance()

        self.setWindowTitle("NexusSearch Desktop")
        self.resize(self.settings.window_width, self.settings.window_height)
        self.setMinimumSize(1260, 820)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(22, 26, 22, 26)
        sidebar_layout.setSpacing(16)

        brand = QLabel("NexusSearch")
        brand.setStyleSheet("font-size: 26px; font-weight: 800; color: #fff4ea;")
        tagline = QLabel("Local-first RAG desktop cockpit")
        tagline.setWordWrap(True)
        tagline.setStyleSheet("color: rgba(255, 241, 226, 0.78); font-size: 13px;")
        sidebar_layout.addWidget(brand)
        sidebar_layout.addWidget(tagline)
        sidebar_layout.addSpacing(10)

        self.nav_buttons: list[NavButton] = []
        for text in ["问答台", "知识库", "设置"]:
            button = NavButton(text)
            sidebar_layout.addWidget(button)
            self.nav_buttons.append(button)
        sidebar_layout.addStretch(1)

        footer = QLabel("后续可直接配合 PyInstaller 打包为 exe。")
        footer.setWordWrap(True)
        footer.setStyleSheet("color: rgba(255, 241, 226, 0.72); font-size: 12px;")
        sidebar_layout.addWidget(footer)

        self.stack = QStackedWidget()
        self.search_page = SearchPage()
        self.library_page = LibraryPage()
        self.settings_page = SettingsPage()
        self.stack.addWidget(self.search_page)
        self.stack.addWidget(self.library_page)
        self.stack.addWidget(self.settings_page)

        root.addWidget(sidebar)
        root.addWidget(self.stack, 1)

        self.nav_buttons[0].clicked.connect(lambda: self._switch_page(0))
        self.nav_buttons[1].clicked.connect(lambda: self._switch_page(1))
        self.nav_buttons[2].clicked.connect(lambda: self._switch_page(2))
        self.nav_buttons[0].setChecked(True)

        self.search_page.ask_requested.connect(self.run_query)
        self.library_page.build_requested.connect(self.run_ingest)
        self.settings_page.saved.connect(self.save_settings)

        self._load_settings_into_pages()
        self.statusBar().showMessage("桌面端已就绪。")

    def _load_settings_into_pages(self) -> None:
        self.search_page.runtime_hint.setText(
            f"工作流: {self.settings.workflow}  ·  模型: {self.settings.ollama_model}  ·  DB: {self.settings.db_path}"
        )
        self.library_page.load_settings(self.settings)
        self.settings_page.load_settings(self.settings)

    def save_settings(self) -> None:
        self.settings = self.settings_page.collect_settings()
        self.store.save(self.settings)
        self._load_settings_into_pages()
        self.statusBar().showMessage("设置已保存。")
        QMessageBox.information(self, "NexusSearch", "设置已保存，下次启动会自动加载。")

    def run_ingest(self, source: str, overwrite: bool) -> None:
        self.settings = self._prepare_settings(source_override=source)
        self.library_page.set_busy(True)
        self.library_page.reset_progress()
        self.library_page.append_log("正在扫描目录并构建索引，请稍候...")

        task = StreamTask(
            lambda: self._make_service().ingest_stream(source=source, overwrite=overwrite)
        )
        task.signals.event.connect(self._handle_ingest_event)
        task.signals.error.connect(self._show_error)
        task.signals.finished.connect(lambda: self.library_page.set_busy(False))
        self.thread_pool.start(task)

    def run_query(self, query: str) -> None:
        self.settings = self._prepare_settings()
        self.search_page.clear_outputs()
        self.search_page.set_busy(True)
        self.statusBar().showMessage("正在检索与生成...")

        task = StreamTask(
            lambda: self._make_service().stream_answer(
                query=query,
                top_k=self.settings.top_k,
                candidate_k=self.settings.candidate_k,
                use_rerank=self.settings.enable_rerank,
                generate=self.settings.generate,
                workflow=self.settings.workflow,
            )
        )
        task.signals.event.connect(self._handle_stream_event)
        task.signals.error.connect(self._show_error)
        task.signals.finished.connect(self._on_stream_finished)
        self.thread_pool.start(task)

    def _prepare_settings(self, source_override: str | None = None) -> DesktopSettings:
        self.settings = self.settings_page.collect_settings()
        if source_override is not None:
            self.settings.source_path = source_override
        self.store.save(self.settings)
        self._load_settings_into_pages()
        return self.settings

    def _handle_stream_event(self, payload: dict[str, Any]) -> None:
        event = payload["event"]
        data = payload["data"]
        if event == "retrieval":
            self.search_page.set_trace(data.get("trace", []))
        elif event == "context":
            self.search_page.add_context(data)
        elif event == "token":
            self.search_page.append_token(data.get("text", ""))
        elif event == "final":
            self.search_page.set_final_answer(data)

    def _on_stream_finished(self) -> None:
        self.search_page.set_busy(False)
        self.statusBar().showMessage("回答完成。")

    def _on_ingest_result(self, summary: dict[str, Any]) -> None:
        self.library_page.show_summary(summary)
        self.statusBar().showMessage("知识库构建完成。")

    def _handle_ingest_event(self, payload: dict[str, Any]) -> None:
        event = payload["event"]
        data = payload["data"]
        if event == "start":
            self.library_page.begin_progress(int(data.get("total_files", 0)))
            self.library_page.append_log(
                f"source={data.get('source')} total_files={data.get('total_files')} overwrite={data.get('overwrite')}"
            )
        elif event == "progress":
            self.library_page.update_progress(
                current=int(data.get("current", 0)),
                total=int(data.get("total", 0)),
                file_path=str(data.get("file_path", "")),
                chunks_in_file=int(data.get("chunks_in_file", 0)),
            )
            self.library_page.append_log(
                f"{data.get('current')}/{data.get('total')} {data.get('file_path')} -> +{data.get('chunks_in_file')} chunks"
            )
        elif event == "final":
            self._on_ingest_result(data)

    def _show_error(self, trace: str) -> None:
        self.search_page.set_busy(False)
        self.library_page.set_busy(False)
        self.statusBar().showMessage("操作失败，请检查错误信息。")
        QMessageBox.critical(self, "NexusSearch Error", trace)

    def _switch_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for idx, button in enumerate(self.nav_buttons):
            button.setChecked(idx == index)

    def _make_service(self) -> NexusRAGService:
        return NexusRAGService(self.settings.to_nexus_config())

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self.settings.window_width = self.width()
        self.settings.window_height = self.height()
        self.store.save(self.settings)
        super().closeEvent(event)


def launch_desktop() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("NexusSearch Desktop")
    app.setStyleSheet(APP_STYLESHEET)
    app.setFont(QFont("Segoe UI Variable", 10))
    window = NexusDesktopWindow()
    window.show()
    return app.exec()
