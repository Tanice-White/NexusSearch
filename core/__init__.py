__all__ = ["NexusConfig", "NexusRAGService"]


def __getattr__(name: str):  # noqa: ANN201
    if name == "NexusConfig":
        from core.config import NexusConfig

        return NexusConfig
    if name == "NexusRAGService":
        from core.service import NexusRAGService

        return NexusRAGService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
