"""LangGraph memory helpers."""

from __future__ import annotations


def build_memory_saver():
    """Return an in-memory LangGraph checkpointer.

    LangGraph's Python docs use `InMemorySaver`; the project requirement here
    wants a `MemorySaver`-style capability, so this helper provides the
    thread-level persistence entrypoint without forcing eager imports at module
    import time.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()
