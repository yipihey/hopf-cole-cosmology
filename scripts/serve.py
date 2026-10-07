#!/usr/bin/env python3
"""Static server for local development that disables caching (so rebuilt WASM is picked up)."""
import http.server, sys, functools
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def log_message(self, *a): pass
H.extensions_map.update({".wasm": "application/wasm", ".js": "text/javascript"})
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
root = sys.argv[2] if len(sys.argv) > 2 else "_site"
http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory=root)).serve_forever()
