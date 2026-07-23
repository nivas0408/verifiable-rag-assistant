import subprocess
import sys
import os
import time
import http.server
import socketserver

class CrashReportHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        content = "No crash log found."
        if os.path.exists("uvicorn_crash.log"):
            with open("uvicorn_crash.log", "r", errors="ignore") as f:
                content = f.read()
        self.wfile.write(f"FastAPI Startup Crash Log:\n\n{content}".encode("utf-8"))

def serve_crash_report(port):
    print(f"Starting crash report server on port {port}...")
    handler = CrashReportHandler
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", int(port)), handler) as httpd:
            httpd.serve_forever()
    except Exception as e:
        print(f"Failed to start crash report server: {e}")

def main():
    port = os.getenv("PORT", "8000")
    print(f"Starting FastAPI backend publicly on 0.0.0.0:{port}...")
    
    # Write stdout/stderr to uvicorn_crash.log
    log_file = open("uvicorn_crash.log", "w", encoding="utf-8")
    
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0", "--port", port
    ], stdout=log_file, stderr=subprocess.STDOUT)

    # Monitor startup for 12 seconds
    for _ in range(12):
        time.sleep(1)
        api_exit = api_process.poll()
        if api_exit is not None:
            log_file.close()
            print(f"FastAPI backend crashed on startup with code {api_exit}.")
            # Serve the crash report on the public port
            serve_crash_report(port)
            sys.exit(api_exit)

    # If it survived startup, close the log file and let it run
    log_file.close()
    
    # Start Streamlit frontend internally on 127.0.0.1:8501
    print("Starting Streamlit frontend internally on 127.0.0.1:8501...")
    env = os.environ.copy()
    env["BACKEND_URL"] = f"http://127.0.0.1:{port}"
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", "8501",
        "--server.address", "127.0.0.1"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Monitor processes
    try:
        while True:
            api_exit = api_process.poll()
            streamlit_exit = streamlit_process.poll()

            if api_exit is not None:
                print(f"FastAPI backend exited unexpectedly with code {api_exit}.")
                streamlit_process.terminate()
                sys.exit(api_exit)
                
            if streamlit_exit is not None:
                print(f"Streamlit frontend exited unexpectedly with code {streamlit_exit}.")
                api_process.terminate()
                sys.exit(streamlit_exit)
                
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down processes gracefully...")
        api_process.terminate()
        streamlit_process.terminate()

if __name__ == "__main__":
    main()
