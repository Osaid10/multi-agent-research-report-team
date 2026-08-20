"""The specialist agents. One job each, one prompt each."""

from . import critic, planner, researcher, supervisor, writer

__all__ = ["critic", "planner", "researcher", "supervisor", "writer"]
