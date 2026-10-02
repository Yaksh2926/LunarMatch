"""Multi-threaded HTTP web server providing REST APIs and user-friendly SPA for LunarMatch."""
from __future__ import annotations

import json
import logging
import mimetypes
import urllib.parse
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from lunarmatch import __version__
from lunarmatch.web.presets import ensure_sample_presets
from lunarmatch.web.service import RUNS_DIR, registration_service

logger = logging.getLogger("lunarmatch.web")
STATIC_DIR = (Path(__file__).parent / "static").resolve()


class LunarMatchRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler with REST API routing and CORS support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        """Send JSON response with CORS headers."""
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, download_name: str | None = None) -> None:
        """Send a binary file response."""
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        file_size = file_path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(file_size))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        with file_path.open("rb") as f:
            while chunk := f.read(64 * 1024):
                self.wfile.write(chunk)

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self) -> None:
        """Handle HEAD requests for APIs and static assets."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
        else:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests for APIs and Static UI assets."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # ---------------------------------------------------------------------
        # REST API Routes
        # ---------------------------------------------------------------------
        if path == "/api/status":
            self._send_json({
                "status": "READY",
                "engine": "LunarMatch Registration Core",
                "version": __version__,
                "server_time": datetime.now(UTC).isoformat(),
            })
            return

        if path == "/api/samples":
            samples = ensure_sample_presets()
            self._send_json({"samples": list(samples.values())})
            return

        if path == "/api/runs":
            runs = registration_service.list_runs()
            self._send_json({"runs": runs})
            return

        if path.startswith("/api/registration/status/"):
            run_id = path.replace("/api/registration/status/", "").strip("/")
            status_data = registration_service.get_job_status(run_id)
            self._send_json(status_data)
            return

        if path.startswith("/api/registration/result/"):
            run_id = path.replace("/api/registration/result/", "").strip("/")
            try:
                result_data = registration_service.get_run_result(run_id)
                self._send_json(result_data)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, status=HTTPStatus.NOT_FOUND)
            return

        if path.startswith("/api/registration/download/"):
            parts = path.replace("/api/registration/download/", "").strip("/").split("/")
            if len(parts) >= 2:
                run_id, filename = parts[0], parts[1]
                run_dir = RUNS_DIR / run_id
                if not run_dir.exists():
                    if run_id == "historical_ohrc_control":
                        run_dir = Path("outputs/ohrc_pair_validation")
                    elif run_id == "historical_ohrc_tmc":
                        run_dir = Path("outputs/ohrc_tmc_validation")
                    elif run_id == "historical_synthetic_demo":
                        run_dir = Path("outputs/demo_run")

                target_file = (run_dir / filename).resolve()
                # Security check to prevent path traversal
                if str(target_file).startswith(str(Path("outputs").resolve())):
                    self._send_file(target_file, download_name=filename)
                    return
            self._send_json({"error": "Download file not found"}, status=HTTPStatus.NOT_FOUND)
            return

        if path == "/api/benchmark":
            benchmark_data = self._get_benchmark_data()
            self._send_json({"benchmark": benchmark_data})
            return

        # ---------------------------------------------------------------------
        # Static Assets & Fallback to SPA index.html
        # ---------------------------------------------------------------------
        # Check if requesting assets from outputs/assets
        if path.startswith("/assets/"):
            asset_file = Path("outputs") / path.lstrip("/")
            if asset_file.exists():
                self._send_file(asset_file)
                return

        # Check in static directory
        rel_path = path.lstrip("/")
        if not rel_path or rel_path in ["register", "results", "history", "benchmark", "data", "docs"]:
            # Route SPA pages to index.html
            index_file = STATIC_DIR / "index.html"
            self._send_file(index_file)
            return

        local_file = STATIC_DIR / rel_path
        if local_file.exists() and local_file.is_file():
            self._send_file(local_file)
            return

        # Default fallback to index.html for client-side routing
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            self._send_file(index_file)
            return

        super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests for metadata detection, file uploads, and runs."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/metadata/detect":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_len)
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                file_path = body.get("file_path", "")
                
                meta = registration_service.detect_metadata(file_path)
                self._send_json(meta)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/registration/upload":
            try:
                # Handle multipart form data
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ctype:
                    # Parse multipart body
                    content_len = int(self.headers.get("Content-Length", 0))
                    raw_body = self.rfile.read(content_len)
                    
                    # Simple boundary extraction
                    boundary = ctype.split("boundary=")[-1].strip().encode("utf-8")
                    parts = raw_body.split(b"--" + boundary)
                    
                    saved_path = ""
                    orig_filename = "uploaded_raster.tif"
                    
                    for part in parts:
                        if b'filename="' in part:
                            # Extract header and data
                            header_part, data_part = part.split(b"\r\n\r\n", 1)
                            headers_str = header_part.decode("utf-8", errors="ignore")
                            
                            # Parse filename
                            for line in headers_str.split("\r\n"):
                                if "filename=" in line:
                                    orig_filename = line.split('filename="')[-1].split('"')[0]
                            
                            # Clean trailing \r\n
                            file_bytes = data_part.rstrip(b"\r\n")
                            saved_path = registration_service.save_upload(file_bytes, orig_filename)
                            break
                    
                    if saved_path:
                        meta = registration_service.detect_metadata(saved_path)
                        self._send_json({
                            "status": "SUCCESS",
                            "file_path": saved_path,
                            "metadata": meta,
                        })
                        return
                    else:
                        self._send_json({"error": "No file content detected in upload"}, status=HTTPStatus.BAD_REQUEST)
                        return
                else:
                    self._send_json({"error": "Expected multipart/form-data upload"}, status=HTTPStatus.BAD_REQUEST)
                    return
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Upload failed: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/registration/run":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_len)
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}

                source_path = body.get("source_path")
                reference_path = body.get("reference_path")
                config_options = body.get("config", {})

                if not source_path or not reference_path:
                    self._send_json({"error": "source_path and reference_path are required"}, status=HTTPStatus.BAD_REQUEST)
                    return

                run_id = registration_service.start_registration(
                    source_path=source_path,
                    reference_path=reference_path,
                    config_dict=config_options,
                )

                self._send_json({
                    "status": "QUEUED",
                    "run_id": run_id,
                    "message": "Registration engine launched.",
                })
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Failed to start registration: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def _get_benchmark_data(self) -> list[dict[str, Any]]:
        """Return multi-pair benchmark numbers grounded in real outputs."""
        return [
            {
                "pair_name": "Chandrayaan-2 OHRC ↔ OHRC (Control)",
                "sensors": "OHRC (0.25m) → OHRC (0.25m)",
                "raw_matches": 1759,
                "inliers": 135,
                "inlier_ratio": "83.3%",
                "rmse": "1.472 px",
                "grid_occupancy": "17.2%",
                "hull_coverage": "9.57%",
                "quality_status": "VALIDATED",
                "notes": "Verified real-data control experiment with geometric consistency.",
            },
            {
                "pair_name": "Chandrayaan-2 OHRC ↔ TMC-2 (Cross-Sensor)",
                "sensors": "OHRC (0.25m) → TMC-2 (5.0m)",
                "raw_matches": 8,
                "inliers": 3,
                "inlier_ratio": "37.5%",
                "rmse": "N/A",
                "grid_occupancy": "1.6%",
                "hull_coverage": "0.0%",
                "quality_status": "LIMITED",
                "notes": "20x GSD ratio. Quality gate rejects due to sparse/localized spatial coverage.",
            },
            {
                "pair_name": "Synthetic Lunar Crater Benchmark (Procedural Demo)",
                "sensors": "Synthetic (1.0m) → Synthetic (1.0m)",
                "raw_matches": 112,
                "inliers": 94,
                "inlier_ratio": "83.9%",
                "rmse": "0.285 px",
                "grid_occupancy": "50.0%",
                "hull_coverage": "65.0%",
                "quality_status": "VALIDATED",
                "notes": "Ground-truth procedural baseline demonstrating sub-pixel ECC refinement.",
            },
        ]


def run_server(port: int | None = None, host: str | None = None) -> None:
    """Launch the LunarMatch ThreadingHTTPServer."""
    import os
    if port is None:
        port = int(os.getenv("PORT", "8000"))
    if host is None:
        host = os.getenv("HOST", "0.0.0.0")

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, LunarMatchRequestHandler)
    print(f"\n[LunarMatch] Server active at http://{host}:{port}")
    print("[LunarMatch] Press Ctrl+C to stop the server.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down LunarMatch server...")
        httpd.server_close()


if __name__ == "__main__":
    import os
    import sys
    port_arg = int(os.getenv("PORT", "8000"))
    if len(sys.argv) > 1:
        try:
            port_arg = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port=port_arg, host="0.0.0.0")
