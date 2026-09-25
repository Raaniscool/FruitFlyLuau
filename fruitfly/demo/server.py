"""Stdlib HTTP server for the desk-fly demo.

No web framework: the project's runtime dependencies are numpy/scipy/pandas/yaml and
this must not add another. ``http.server`` is sufficient for a single page, two JSON
endpoints and one image.

Routes
------
``GET  /``              the page
``GET  /static/*``      assets
``GET  /api/status``    what the project can actually do right now (measured, not claimed)
``POST /api/generate``  {"prompt": str, "backend": "template"|"connectome"} -> Response
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..utils import get_logger
from ..version import __version__
from .backends import BACKENDS, get_backend

log = get_logger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
MAX_PROMPT = 2000


def status_payload() -> dict:
    """What is true today. Every field here is a measured fact or an explicit 'no'."""
    return {
        "version": __version__,
        "backends": {
            name: {"available": b.available()[0], "why": b.available()[1]}
            for name, b in BACKENDS.items()
        },
        "learning_demonstrated": False,
        "headline": "framework verified end-to-end, learning not yet demonstrated",
        "measured": {
            "connectome": "FAFB v783: 5,342,446 connection rows -> 3,732,460 pair edges "
                          "over 138,584 neurons",
            "subgraph": "300 of 5,591 mushroom-body neurons, 6,237 edges, 0 isolated",
            "reservoir_real": "memory capacity 0.10, separation 1.13",
            "reservoir_shuffled": "memory capacity 0.14, separation 1.04",
            "verdict": "no measurable advantage for the real wiring over a "
                       "degree-preserving shuffle",
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = f"FruitFlyLuauDemo/{__version__}"

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        log.debug("%s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------------ helpers
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, indent=2).encode("utf-8"), "application/json")

    def _static(self, rel: str) -> None:
        # resolve() + relative_to() so "../" cannot escape the static directory
        target = (STATIC / rel).resolve()
        try:
            target.relative_to(STATIC.resolve())
        except ValueError:
            self._json(403, {"error": "path outside the static directory"})
            return
        if not target.is_file():
            self._json(404, {"error": f"no such asset: {rel}"})
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype)

    # --------------------------------------------------------------------- verbs
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._static("index.html")
        elif path == "/api/status":
            self._json(200, status_payload())
        elif path.startswith("/static/"):
            self._static(path[len("/static/"):])
        else:
            self._json(404, {"error": f"no route {path}"})

    do_HEAD = do_GET

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/generate":
            self._json(404, {"error": f"no route {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return
        if length > 64 * 1024:
            self._json(413, {"error": "request too large"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._json(400, {"error": f"invalid JSON: {exc.msg}"})
            return

        prompt = str(body.get("prompt", ""))[:MAX_PROMPT]
        name = str(body.get("backend", "template"))
        try:
            backend = get_backend(name)
        except KeyError as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(200, backend.generate(prompt).to_dict())


def serve(host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    """Create (but do not start) the server, so callers control the loop."""
    httpd = ThreadingHTTPServer((host, port), Handler)
    log.info("demo server on http://%s:%d", host, port)
    return httpd


def run(host: str = "0.0.0.0", port: int = 8000) -> int:
    httpd = serve(host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        httpd.server_close()
    return 0
