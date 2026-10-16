"""Evaluator-owned rerun broker: localiser receives an opaque URL, never --bug paths."""
from __future__ import annotations
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading

from reference.format import OPS


@contextmanager
def replay_broker(commands: dict[str, list[str]], logs: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            ticket = self.path.strip('/')
            try:
                if ticket not in commands:
                    raise ValueError('unknown ticket')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536:
                    raise ValueError('invalid request length')
                body = json.loads(self.rfile.read(length))
                argv = commands[ticket] + ['--dump', body['dump']]
                for patch in body['patches']:
                    op, path = patch.split('=', 1)
                    if op not in OPS or not Path(path).is_file():
                        raise ValueError('invalid patch')
                    argv += ['--patch', patch]
                if body.get('patch_ids'):
                    argv += ['--patch-ids', body['patch_ids']]
                result = subprocess.run(argv, capture_output=True, text=True, timeout=300)
                with (logs / f'{ticket}.log').open('a') as f:
                    f.write(result.stdout + result.stderr)
                ok = result.returncode == 0
            except Exception:
                ok = False
            response = json.dumps(dict(ok=ok)).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
