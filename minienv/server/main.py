import http.server
import json
import os
import socketserver
import subprocess
import threading
import base64

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
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                command = data.get("command", [])
                timeout = data.get("timeout", 90)
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
                response = {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
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
        elif self.path == "/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                destination = data.get("destination")
                content_b64 = data.get("content")
                if not destination or content_b64 is None:
                    raise ValueError("Missing 'destination' or 'content'")
                file_bytes = base64.b64decode(content_b64)
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with open(destination, "wb") as f:
                    f.write(file_bytes)
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == "/download":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                source = data.get("source")
                if not source:
                    raise ValueError("Missing 'source'")
                with open(source, "rb") as f:
                    file_bytes = f.read()
                content_b64 = base64.b64encode(file_bytes).decode("utf-8")
                response = {"content": content_b64}
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
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "shutting down"}).encode())
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
