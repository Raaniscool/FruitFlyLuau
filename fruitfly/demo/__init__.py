"""Desk-fly demo: a user interface, explicitly not a scientific result.

See ``backends.py`` for the boundary between the placeholder snippet library and the
(not yet existing) connectome-driven generator.
"""

from .backends import BACKENDS, Response, get_backend
from .server import run, serve, status_payload

__all__ = ["BACKENDS", "Response", "get_backend", "run", "serve", "status_payload"]
