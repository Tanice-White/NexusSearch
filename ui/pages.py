from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.settings_store import DesktopSettings


def build_card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("Card")
    shadow = QGraphicsDropShadowEffect(frame)
    shadow.setBlurRadius(32)
    shadow.setOffset(0, 14)
    shadow.setColor(QColor(33, 24, 15, 30))
    frame.setGraphicsEffect(shadow)
    return frame


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


class NavButton(QPushButton):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setObjectName("GhostButton")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)


class SearchPage(QWidget):
    ask_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.context_payloads: list[dict[str, Any]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(18)

        hero = build_card()
        hero_layout = QVBoxLayout(hero)
        title = QLabel("本地知识搜索，像 Perplexity 一样优雅。")
        title.setObjectName("HeroTitle")
        subtitle = QLabel("直接提问，流式展示答案、引用片段与 Self-RAG 轨迹。")
        subtitle.setObjectName("HeroSubTitle")
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)

        composer = build_card()
        composer_layout = QVBoxLayout(composer)
        header = QHBoxLayout()
        header.addWidget(section_label("提问"))
        header.addStretch(1)
        clear_button = QPushButton("清空结果")
        clear_button.clicked.connect(self.clear_outputs)
        self.ask_button = QPushButton("开始询问")
        self.ask_button.setObjectName("PrimaryButton")
        self.ask_button.clicked.connect(self._emit_query)
        header.addWidget(clear_button)
        header.addWidget(self.ask_button)

        self.query_input = QTextEdit()
        self.query_input.setFixedHeight(140)
        self.query_input.setPlaceholderText("例如：这个系统如何用 RRF 统一 Dense 与 Sparse 的检索分数？")
        self.runtime_hint = QLabel("当前会使用设置页里的模型、检索和工作流配置。")
        self.runtime_hint.setObjectName("HeroSubTitle")

        composer_layout.addLayout(header)
        composer_layout.addWidget(self.query_input)
        composer_layout.addWidget(self.runtime_hint)

        content = QHBoxLayout()
        content.setSpacing(18)

        answer_card = build_card()
        answer_layout = QVBoxLayout(answer_card)
        answer_layout.addWidget(section_label("流式回答"))
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setPlaceholderText("回答会在这里逐字流式出现。")
        answer_layout.addWidget(self.answer_output)

        source_card = build_card()
        source_layout = QVBoxLayout(source_card)
        source_layout.addWidget(section_label("引用片段"))
        self.source_list = QListWidget()
        self.source_list.currentRowChanged.connect(self._show_source_preview)
        self.source_preview = QTextEdit()
        self.source_preview.setReadOnly(True)
        self.source_preview.setPlaceholderText("点击左侧片段查看全文。")
        source_layout.addWidget(self.source_list, 3)
        source_layout.addWidget(self.source_preview, 2)

        content.addWidget(answer_card, 8)
        content.addWidget(source_card, 5)

        trace_card = build_card()
        trace_layout = QVBoxLayout(trace_card)
        trace_layout.addWidget(section_label("工作流轨迹"))
        self.trace_output = QTextEdit()
        self.trace_output.setReadOnly(True)
        self.trace_output.setFixedHeight(170)
        trace_layout.addWidget(self.trace_output)

        root.addWidget(hero)
        root.addWidget(composer)
        root.addLayout(content, 1)
        root.addWidget(trace_card)

    def _emit_query(self) -> None:
        query = self.query_input.toPlainText().strip()
        if query:
            self.ask_requested.emit(query)

    def clear_outputs(self) -> None:
        self.answer_output.clear()
        self.trace_output.clear()
        self.source_list.clear()
        self.source_preview.clear()
        self.context_payloads.clear()

    def append_token(self, text: str) -> None:
        self.answer_output.moveCursor(QTextCursor.End)
        self.answer_output.insertPlainText(text)
        self.answer_output.ensureCursorVisible()

    def set_trace(self, trace: list[dict[str, Any]]) -> None:
        lines = []
        for step in trace:
            label = step.get("step", "unknown")
            payload = ", ".join(
                f"{key}={value}" for key, value in step.items() if key != "step"
            )
            lines.append(f"{label}: {payload}")
        self.trace_output.setPlainText("\n".join(lines))

    def add_context(self, payload: dict[str, Any]) -> None:
        self.context_payloads.append(payload)
        item = QListWidgetItem(f"{payload['header_path']}  ·  {payload['source_path']}")
        item.setToolTip(payload["text"])
        self.source_list.addItem(item)

    def set_final_answer(self, payload: dict[str, Any]) -> None:
        self.answer_output.setPlainText(payload["answer"])
        self.set_trace(payload.get("trace", []))
        self.source_list.clear()
        self.context_payloads = []
        for context in payload.get("contexts", []):
            self.add_context(context)
        if self.source_list.count() > 0:
            self.source_list.setCurrentRow(0)

    def set_busy(self, busy: bool) -> None:
        self.ask_button.setDisabled(busy)
        self.ask_button.setText("生成中..." if busy else "开始询问")

    def _show_source_preview(self, row: int) -> None:
        if row < 0 or row >= len(self.context_payloads):
            self.source_preview.clear()
            return
        payload = self.context_payloads[row]
        self.source_preview.setPlainText(
            f"来源: {payload['source_path']}\n"
            f"标题: {payload['header_path']}\n"
            f"分数: {payload.get('score', 0):.4f}\n\n"
            f"{payload['text']}"
        )


class LibraryPage(QWidget):
    build_requested = Signal(str, bool)

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(18)

        hero = build_card()
        hero_layout = QVBoxLayout(hero)
        title = QLabel("知识库建库")
        title.setObjectName("HeroTitle")
        subtitle = QLabel("选择本地目录，扫描文档与代码文件，完成本地索引构建。")
        subtitle.setObjectName("HeroSubTitle")
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)

        controls = build_card()
        controls_layout = QVBoxLayout(controls)
        path_row = QHBoxLayout()
        self.source_path_input = QLineEdit()
        browse = QPushButton("浏览路径")
        browse.clicked.connect(self._browse_source)
        path_row.addWidget(self.source_path_input, 1)
        path_row.addWidget(browse)

        db_row = QHBoxLayout()
        self.db_path_input = QLineEdit()
        self.db_path_input.setReadOnly(True)
        db_row.addWidget(self.db_path_input, 1)

        actions = QHBoxLayout()
        self.overwrite_checkbox = QCheckBox("重建索引时覆盖旧表")
        self.overwrite_checkbox.setChecked(True)
        self.build_button = QPushButton("开始建库")
        self.build_button.setObjectName("PrimaryButton")
        self.build_button.clicked.connect(self._emit_build)
        actions.addWidget(self.overwrite_checkbox)
        actions.addStretch(1)
        actions.addWidget(self.build_button)

        controls_layout.addWidget(section_label("扫描源目录"))
        controls_layout.addLayout(path_row)
        controls_layout.addWidget(section_label("当前数据库目录"))
        controls_layout.addLayout(db_row)
        controls_layout.addLayout(actions)

        progress_card = build_card()
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.addWidget(section_label("建库进度"))
        self.progress_label = QLabel("等待开始")
        self.progress_label.setObjectName("HeroSubTitle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)

        metrics_card = build_card()
        metrics_layout = QGridLayout(metrics_card)
        self.metric_files = self._metric_card("扫描文件")
        self.metric_chunks = self._metric_card("切块数量")
        self.metric_indexed = self._metric_card("写入记录")
        self.metric_strategy = self._metric_card("切块策略")
        metrics_layout.addWidget(self.metric_files[0], 0, 0)
        metrics_layout.addWidget(self.metric_chunks[0], 0, 1)
        metrics_layout.addWidget(self.metric_indexed[0], 0, 2)
        metrics_layout.addWidget(self.metric_strategy[0], 0, 3)

        log_card = build_card()
        log_layout = QVBoxLayout(log_card)
        log_layout.addWidget(section_label("建库日志"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("建库完成后会在这里显示摘要。")
        log_layout.addWidget(self.log_output)

        files_card = build_card()
        files_layout = QVBoxLayout(files_card)
        files_layout.addWidget(section_label("扫描中文件"))
        self.files_list = QListWidget()
        self.files_list.setAlternatingRowColors(False)
        files_layout.addWidget(self.files_list)

        root.addWidget(hero)
        root.addWidget(controls)
        root.addWidget(progress_card)
        root.addWidget(metrics_card)
        bottom = QHBoxLayout()
        bottom.setSpacing(18)
        bottom.addWidget(log_card, 4)
        bottom.addWidget(files_card, 3)
        root.addLayout(bottom, 1)

    def load_settings(self, settings: DesktopSettings) -> None:
        self.source_path_input.setText(settings.source_path)
        self.db_path_input.setText(settings.db_path)

    def set_busy(self, busy: bool) -> None:
        self.build_button.setDisabled(busy)
        self.build_button.setText("建库中..." if busy else "开始建库")
        if not busy and self.progress_bar.value() < 100:
            self.progress_label.setText("等待开始")

    def append_log(self, text: str) -> None:
        self.log_output.append(text)

    def show_summary(self, summary: dict[str, Any]) -> None:
        self.metric_files[1].setText(str(summary.get("files", "--")))
        self.metric_chunks[1].setText(str(summary.get("chunks", "--")))
        self.metric_indexed[1].setText(str(summary.get("indexed", "--")))
        self.metric_strategy[1].setText(str(summary.get("chunking_strategy", "--")))
        self.log_output.setPlainText("\n".join(f"{key}: {value}" for key, value in summary.items()))
        self.progress_bar.setValue(100)
        self.progress_label.setText("建库完成")

    def reset_progress(self) -> None:
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备扫描文件...")
        self.files_list.clear()

    def begin_progress(self, total_files: int) -> None:
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"已发现 {total_files} 个文件，开始解析...")

    def update_progress(self, current: int, total: int, file_path: str, chunks_in_file: int) -> None:
        if total > 0:
            self.progress_bar.setValue(int(current * 100 / total))
        else:
            self.progress_bar.setValue(0)
        self.progress_label.setText(f"{current}/{total} · +{chunks_in_file} chunks")
        item = QListWidgetItem(f"[{current}/{total}] {file_path}")
        item.setToolTip(file_path)
        self.files_list.addItem(item)
        self.files_list.scrollToBottom()

    def _emit_build(self) -> None:
        source = self.source_path_input.text().strip()
        if source:
            self.build_requested.emit(source, self.overwrite_checkbox.isChecked())

    def _browse_source(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择本地目录",
            self.source_path_input.text() or str(Path.cwd()),
        )
        if selected:
            self.source_path_input.setText(selected)

    @staticmethod
    def _metric_card(label: str) -> tuple[QFrame, QLabel]:
        frame = build_card()
        layout = QVBoxLayout(frame)
        title = QLabel(label)
        title.setObjectName("MetricLabel")
        value = QLabel("--")
        value.setObjectName("MetricValue")
        layout.addWidget(title)
        layout.addWidget(value)
        layout.addStretch(1)
        return frame, value


class SettingsPage(QWidget):
    saved = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.inputs: dict[str, Any] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)
        root.addWidget(scroll)

        layout = QVBoxLayout(container)
        layout.setSpacing(18)

        hero = build_card()
        hero_layout = QVBoxLayout(hero)
        title = QLabel("设置中心")
        title.setObjectName("HeroTitle")
        subtitle = QLabel("模型、检索、切块、数据库路径等配置统一放在这里。")
        subtitle.setObjectName("HeroSubTitle")
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)

        layout.addWidget(hero)
        layout.addWidget(self._storage_group())
        layout.addWidget(self._retrieval_group())
        layout.addWidget(self._model_group())
        layout.addWidget(self._appearance_group())

        action_card = build_card()
        action_layout = QHBoxLayout(action_card)
        action_layout.addWidget(section_label("保存后会立即应用到桌面端。"))
        action_layout.addStretch(1)
        self.save_button = QPushButton("保存设置")
        self.save_button.setObjectName("PrimaryButton")
        self.save_button.clicked.connect(self.saved.emit)
        action_layout.addWidget(self.save_button)
        layout.addWidget(action_card)
        layout.addStretch(1)

    def load_settings(self, settings: DesktopSettings) -> None:
        self.inputs["source_path"].setText(settings.source_path)
        self.inputs["db_path"].setText(settings.db_path)
        self.inputs["table_name"].setText(settings.table_name)
        self.inputs["chunking_strategy"].setCurrentText(settings.chunking_strategy)
        self.inputs["chunk_size"].setValue(settings.chunk_size)
        self.inputs["chunk_overlap"].setValue(settings.chunk_overlap)
        self.inputs["semantic_similarity_threshold"].setValue(settings.semantic_similarity_threshold)
        self.inputs["semantic_min_sentences"].setValue(settings.semantic_min_sentences)
        self.inputs["top_k"].setValue(settings.top_k)
        self.inputs["candidate_k"].setValue(settings.candidate_k)
        self.inputs["enable_rerank"].setChecked(settings.enable_rerank)
        self.inputs["workflow"].setCurrentText(settings.workflow)
        self.inputs["generate"].setChecked(settings.generate)
        self.inputs["self_rag_max_loops"].setValue(settings.self_rag_max_loops)
        self.inputs["prefer_langgraph"].setChecked(settings.prefer_langgraph)
        self.inputs["embedding_model"].setText(settings.embedding_model)
        self.inputs["embedding_device"].setCurrentText(settings.embedding_device)
        self.inputs["reranker_model"].setText(settings.reranker_model)
        self.inputs["ollama_base_url"].setText(settings.ollama_base_url)
        self.inputs["ollama_model"].setText(settings.ollama_model)
        self.inputs["theme_mode"].setCurrentText(settings.theme_mode)

    def collect_settings(self) -> DesktopSettings:
        return DesktopSettings(
            source_path=self.inputs["source_path"].text().strip() or "data",
            db_path=self.inputs["db_path"].text().strip() or "data/lancedb",
            table_name=self.inputs["table_name"].text().strip() or "nexus_chunks",
            chunking_strategy=self.inputs["chunking_strategy"].currentText(),
            chunk_size=self.inputs["chunk_size"].value(),
            chunk_overlap=self.inputs["chunk_overlap"].value(),
            semantic_similarity_threshold=self.inputs["semantic_similarity_threshold"].value(),
            semantic_min_sentences=self.inputs["semantic_min_sentences"].value(),
            top_k=self.inputs["top_k"].value(),
            candidate_k=self.inputs["candidate_k"].value(),
            enable_rerank=self.inputs["enable_rerank"].isChecked(),
            workflow=self.inputs["workflow"].currentText(),
            generate=self.inputs["generate"].isChecked(),
            self_rag_max_loops=self.inputs["self_rag_max_loops"].value(),
            prefer_langgraph=self.inputs["prefer_langgraph"].isChecked(),
            embedding_model=self.inputs["embedding_model"].text().strip() or "BAAI/bge-m3",
            embedding_device=self.inputs["embedding_device"].currentText(),
            reranker_model=self.inputs["reranker_model"].text().strip() or "BAAI/bge-reranker-v2-m3",
            ollama_base_url=self.inputs["ollama_base_url"].text().strip() or "http://localhost:11434",
            ollama_model=self.inputs["ollama_model"].text().strip() or "qwen2.5:7b-instruct",
            theme_mode=self.inputs["theme_mode"].currentText(),
        )

    def _storage_group(self) -> QGroupBox:
        group = QGroupBox("存储与建库")
        layout = QFormLayout(group)
        self.inputs["source_path"] = QLineEdit()
        self.inputs["db_path"] = QLineEdit()
        self.inputs["table_name"] = QLineEdit()
        self.inputs["chunking_strategy"] = QComboBox()
        self.inputs["chunking_strategy"].addItems(["semantic", "header"])
        self.inputs["chunk_size"] = QSpinBox()
        self.inputs["chunk_size"].setRange(128, 4096)
        self.inputs["chunk_size"].setSingleStep(64)
        self.inputs["chunk_overlap"] = QSpinBox()
        self.inputs["chunk_overlap"].setRange(0, 1024)
        self.inputs["semantic_similarity_threshold"] = QDoubleSpinBox()
        self.inputs["semantic_similarity_threshold"].setRange(0.1, 0.99)
        self.inputs["semantic_similarity_threshold"].setSingleStep(0.01)
        self.inputs["semantic_min_sentences"] = QSpinBox()
        self.inputs["semantic_min_sentences"].setRange(1, 12)
        layout.addRow("默认扫描目录", self.inputs["source_path"])
        layout.addRow("向量数据库目录", self.inputs["db_path"])
        layout.addRow("数据表名", self.inputs["table_name"])
        layout.addRow("切块策略", self.inputs["chunking_strategy"])
        layout.addRow("Chunk Size", self.inputs["chunk_size"])
        layout.addRow("Chunk Overlap", self.inputs["chunk_overlap"])
        layout.addRow("语义切分阈值", self.inputs["semantic_similarity_threshold"])
        layout.addRow("最少句子数", self.inputs["semantic_min_sentences"])
        return group

    def _retrieval_group(self) -> QGroupBox:
        group = QGroupBox("检索与工作流")
        layout = QFormLayout(group)
        self.inputs["top_k"] = QSpinBox()
        self.inputs["top_k"].setRange(1, 50)
        self.inputs["candidate_k"] = QSpinBox()
        self.inputs["candidate_k"].setRange(1, 200)
        self.inputs["enable_rerank"] = QCheckBox("启用 Cross-Encoder 精排")
        self.inputs["workflow"] = QComboBox()
        self.inputs["workflow"].addItems(["self-rag", "vanilla"])
        self.inputs["generate"] = QCheckBox("启用本地生成")
        self.inputs["self_rag_max_loops"] = QSpinBox()
        self.inputs["self_rag_max_loops"].setRange(0, 6)
        self.inputs["prefer_langgraph"] = QCheckBox("优先使用 LangGraph")
        layout.addRow("最终返回数量", self.inputs["top_k"])
        layout.addRow("候选召回池", self.inputs["candidate_k"])
        layout.addRow("", self.inputs["enable_rerank"])
        layout.addRow("默认工作流", self.inputs["workflow"])
        layout.addRow("", self.inputs["generate"])
        layout.addRow("Self-RAG 最大循环", self.inputs["self_rag_max_loops"])
        layout.addRow("", self.inputs["prefer_langgraph"])
        return group

    def _model_group(self) -> QGroupBox:
        group = QGroupBox("模型与推理")
        layout = QFormLayout(group)
        self.inputs["embedding_model"] = QLineEdit()
        self.inputs["embedding_device"] = QComboBox()
        self.inputs["embedding_device"].addItems(["cpu", "cuda"])
        self.inputs["reranker_model"] = QLineEdit()
        self.inputs["ollama_base_url"] = QLineEdit()
        self.inputs["ollama_model"] = QLineEdit()
        layout.addRow("Embedding 模型", self.inputs["embedding_model"])
        layout.addRow("Embedding 设备", self.inputs["embedding_device"])
        layout.addRow("Reranker 模型", self.inputs["reranker_model"])
        layout.addRow("Ollama 地址", self.inputs["ollama_base_url"])
        layout.addRow("Ollama 模型", self.inputs["ollama_model"])
        return group

    def _appearance_group(self) -> QGroupBox:
        group = QGroupBox("外观与体验")
        layout = QFormLayout(group)
        self.inputs["theme_mode"] = QComboBox()
        self.inputs["theme_mode"].addItems(["sunset", "forest", "linen"])
        layout.addRow("界面主题", self.inputs["theme_mode"])
        return group
