import http.server
import json
import os
import socketserver
import subprocess
import threading


class ExecHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/exec":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                command = data.get("command", [])

                result = subprocess.run(command, capture_output=True, text=True, timeout=30)

                response = {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == "/shutdown":
            # Respond before shutting down
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "shutting down"}).encode())
            # Shutdown in a new thread to avoid blocking the handler
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()


def main():
    port_env = os.environ.get("MINIENV_PORT")
    if port_env is None:
        raise RuntimeError("MINIENV_PORT environment variable not set")
    try:
        port = int(port_env)
    except ValueError:
        raise RuntimeError("MINIENV_PORT must be an integer")

    with socketserver.TCPServer(("", port), ExecHandler) as httpd:
        print(f"Server running on port {port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
