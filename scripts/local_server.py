from __future__ import annotations

import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amfd.backend.service import DiagnosisRequest, DiagnosisService  # noqa: E402

WEB_ROOT = ROOT / "web"
SERVICE = DiagnosisService()


class LocalHandler(BaseHTTPRequestHandler):
    server_version = "AMFDLocal/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json({"status": "ok", "service": "amfd-local"})
            return
        if parsed.path == "/api/v1/demo":
            query = parse_qs(parsed.query)
            machine_id = query.get("machine_id", ["PUMP-101"])[0]
            self._json(SERVICE.demo(machine_id).model_dump(mode="json"))
            return
        if parsed.path == "/":
            self._file(WEB_ROOT / "index.html")
            return
        self._file(WEB_ROOT / parsed.path.lstrip("/"))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        if parsed.path == "/api/v1/diagnose":
            try:
                payload = json.loads(body)
                response = SERVICE.diagnose_window(DiagnosisRequest(**payload))
                self._json(response.model_dump(mode="json"))
            except Exception as exc:  # noqa: BLE001 - HTTP boundary returns a clean error envelope.
                self._json({"detail": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/v1/diagnose/csv":
            query = parse_qs(parsed.query)
            machine_id = query.get("machine_id", ["PUMP-101"])[0]
            notes = query.get("operator_notes", [""])[0]
            force_review = query.get("force_human_review", ["false"])[0].lower() == "true"
            try:
                response = SERVICE.diagnose_csv_text(body, machine_id, notes, force_review)
                self._json(response.model_dump(mode="json"))
            except Exception as exc:  # noqa: BLE001 - HTTP boundary returns a clean error envelope.
                self._json({"detail": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._json({"detail": "Not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", media_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    host = "127.0.0.1"
    port = 8765
    print(f"AMFD local server running at http://{host}:{port}")
    ThreadingHTTPServer((host, port), LocalHandler).serve_forever()


if __name__ == "__main__":
    main()
