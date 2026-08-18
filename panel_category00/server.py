"""
server.py — Minimal HTTP server broadcasting live category00 events to the
panel. Standard library only. Same pattern as this project's other panels.
"""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PANEL_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "phase": "idle",
            "train_progress": 0.0,
            "live_progress": 0.0,
            "current_trial": None,
            "running_accuracy": None,
            "finished": False,
        }


STATE = _State()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            body = PANEL_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            with STATE.lock:
                self._send_json(STATE.data)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        with STATE.lock:
            STATE.data.update(payload)
        self._send_json({"ok": True})


def start(port=8903, open_browser=True):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/"
    print(f"[panel] serving at {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return server


def push(**fields):
    with STATE.lock:
        STATE.data.update(fields)
