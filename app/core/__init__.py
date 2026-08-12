"""UI-neutral assistant core. Front ends depend on this, never the reverse."""
from .assistant import Assistant, AssistantEvent

__all__ = ["Assistant", "AssistantEvent"]
