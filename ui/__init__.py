__all__ = ["launch_desktop"]


def __getattr__(name: str):  # noqa: ANN201
    if name == "launch_desktop":
        from ui.desktop import launch_desktop

        return launch_desktop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
