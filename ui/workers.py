from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class TaskRunner(QRunnable):
    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as exc:  # pragma: no cover - UI path
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.signals.error.emit(trace)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class StreamSignals(QObject):
    event = Signal(object)
    error = Signal(str)
    finished = Signal()


class StreamTask(QRunnable):
    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = StreamSignals()

    def run(self) -> None:
        try:
            for payload in self.fn():
                self.signals.event.emit(payload)
        except Exception as exc:  # pragma: no cover - UI path
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.signals.error.emit(trace)
        finally:
            self.signals.finished.emit()
