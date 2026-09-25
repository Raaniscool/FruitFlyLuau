"""The demo is a user interface, not a result. These tests police that boundary.

Most of them are about honesty rather than functionality: a demo that shows a fly
typing Luau is exactly the thing a casual viewer will mistake for "the fly brain
writes code", so the labelling is a feature with tests, not a comment.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fruitfly.demo.backends import BACKENDS, ConnectomeBackend, TemplateBackend, get_backend
from fruitfly.demo.server import serve, status_payload

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------- backends
def test_every_template_response_is_labelled_a_placeholder():
    b = TemplateBackend()
    for prompt in ("make a part", "kill brick", "banana bread", ""):
        r = b.generate(prompt)
        assert r.is_placeholder is True
        assert r.provenance == "template"
        assert "NOT from the simulated fly brain" in r.notice


def test_the_connectome_backend_refuses_instead_of_falling_back():
    """A silent fallback to the snippet library is how a demo starts lying."""
    r = ConnectomeBackend().generate("make a part")
    assert r.provenance == "unavailable"
    assert r.code == "", "an unavailable backend must not return code from somewhere else"
    assert r.is_placeholder is False
    assert "Not implemented" in r.notice
    for snippet_marker in ("Instance.new", "game:GetService"):
        assert snippet_marker not in r.notice


def test_the_connectome_backend_says_why_and_the_reason_is_the_true_one():
    ok, why = ConnectomeBackend().available()
    assert ok is False
    assert "above chance" in why and "decoder" in why


def test_template_routing_picks_the_snippet_a_human_would_expect():
    b = TemplateBackend()
    cases = {
        "make a kill brick": "kill brick",
        "make a part": "part",
        "tween a part upward": "tween",   # 'part' also matches; first mention wins
        "leaderstats with coins": "leaderstats",
        "a gui button": "gui",
        "teleport the player": "teleport",
    }
    for prompt, expected in cases.items():
        assert b.generate(prompt).matched == expected, prompt


def test_an_unmatched_prompt_admits_it_rather_than_inventing():
    r = TemplateBackend().generate("write me a raytracer in brainfuck")
    assert r.matched == "none (fallback)"
    assert "placeholder" in r.code.lower()


def test_unknown_backend_name_is_rejected():
    with pytest.raises(KeyError):
        get_backend("fly_brain_definitely_real")


def test_status_payload_does_not_overclaim():
    s = status_payload()
    assert s["learning_demonstrated"] is False
    assert "not yet demonstrated" in s["headline"]
    assert s["backends"]["connectome"]["available"] is False
    assert "no measurable advantage" in s["measured"]["verdict"]


# --------------------------------------------------------------------- server
@pytest.fixture()
def live_server():
    httpd = serve("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _post(url: str, payload: dict):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read())


def test_page_carries_the_disclaimer_before_any_javascript_runs(live_server):
    """The banner must be in the served HTML, not painted in later by a script."""
    status, body = _get(live_server + "/")
    html = body.decode()
    assert status == 200
    assert "Placeholder backend" in html
    assert "not from the\n  simulated fly brain" in html.lower() or \
           "not from the simulated fly brain" in " ".join(html.lower().split())
    assert "not a biological claim" in html


def test_generate_endpoint_round_trip(live_server):
    status, data = _post(live_server + "/api/generate", {"prompt": "make a kill brick"})
    assert status == 200
    assert "killBrick.Touched" in data["code"]
    assert data["is_placeholder"] is True


def test_generate_rejects_an_unknown_backend(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(live_server + "/api/generate", {"prompt": "hi", "backend": "nope"})
    assert exc.value.code == 400


def test_static_serving_cannot_escape_its_directory(live_server):
    for attack in ("/static/../server.py", "/static/../../pyproject.toml"):
        try:
            status, _ = _get(live_server + attack)
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status in (403, 404), f"{attack} was served"


def test_the_fly_image_is_actually_shipped(live_server):
    status, body = _get(live_server + "/static/fly_desk.png")
    assert status == 200
    assert body[:8] == b"\x89PNG\r\n\x1a\n"


def test_unknown_route_is_a_clean_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(live_server + "/admin")
    assert exc.value.code == 404


def test_demo_adds_no_third_party_import():
    """The project ships numpy/scipy/pandas/yaml; the demo must not add a web framework."""
    import ast

    for name in ("server.py", "backends.py", "__init__.py"):
        src = (ROOT / "fruitfly" / "demo" / name).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            mod = ""
            if isinstance(node, ast.Import):
                mod = node.names[0].name
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            top = mod.split(".")[0]
            assert top in {
                "", "__future__", "json", "re", "mimetypes", "http", "pathlib", "dataclasses",
                "threading", "fruitfly", "utils", "version", "backends", "server", "typing",
            }, f"{name} imports {mod!r}"
