#!/usr/bin/env python3
"""Serve camera-array visualizations with HTTP byte-range support."""

from __future__ import annotations

import argparse
import html
import io
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class VideoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, landing_page=None, **kwargs):
        self.landing_page = landing_page
        super().__init__(*args, **kwargs)

    def send_head(self):
        self.byte_range = None
        if self.landing_page is not None and urlsplit(self.path).path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.landing_page)))
            self.end_headers()
            return io.BytesIO(self.landing_page)
        requested = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not requested or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
        size = path.stat().st_size
        if match and any(match.groups()) and size:
            first, last = match.groups()
            start = int(first) if first else max(0, size - int(last))
            end = min(int(last), size - 1) if first and last else size - 1
            if 0 <= start <= end < size:
                stream = path.open("rb")
                stream.seek(start)
                self.byte_range = (start, end)
                self.send_response(206)
                self.send_header("Content-Type", self.guess_type(str(path)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
                self.end_headers()
                return stream
        self.send_response(416)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return None

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def copyfile(self, source, outputfile):
        try:
            if self.byte_range is None:
                return super().copyfile(source, outputfile)
            remaining = self.byte_range[1] - self.byte_range[0] + 1
            while remaining:
                block = source.read(min(1 << 20, remaining))
                if not block:
                    break
                outputfile.write(block)
                remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass


def scene_page(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    title = html.escape(str(manifest.get("scene", directory.name)))
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title} · Track4D-Calib</title>
<style>
body{{margin:0;background:#070b11;color:#edf3f7;font:16px system-ui,sans-serif}}
main{{max-width:1440px;margin:auto;padding:32px}}h1{{font-size:24px;font-weight:500}}
video{{width:100%;border-radius:12px;background:#000}}a{{color:#62dbeb;margin-right:24px}}
nav{{margin-top:20px}}
</style>
<main><h1>{title} · Camera array</h1>
<video controls playsinline preload="metadata" poster="preview.png" src="camera_array.mp4"></video>
<nav><a href="camera_array.mp4" download>Download video</a>
<a href="manifest.json">Camera and rendering metadata</a></nav></main></html>
""".encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, required=True
    )
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    landing_page = None
    if not (args.directory / "index.html").is_file():
        if not all((args.directory / name).is_file() for name in ("camera_array.mp4", "manifest.json")):
            parser.error("directory must contain index.html, or camera_array.mp4 and manifest.json")
        landing_page = scene_page(args.directory)
    handler = partial(VideoHandler, directory=str(args.directory.resolve()), landing_page=landing_page)
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(f"Serving {args.directory} on http://{args.bind}:{args.port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
