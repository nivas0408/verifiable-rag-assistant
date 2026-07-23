import subprocess
import sys
import os
import time

def main():
    port = os.getenv("PORT", "8000")
    print(f"Starting FastAPI backend publicly on 0.0.0.0:{port}...")
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0", "--port", port
    ])

    # Give Uvicorn a couple of seconds to bind
    time.sleep(3)

    # Start Streamlit frontend internally on 127.0.0.1:8501
    print("Starting Streamlit frontend internally on 127.0.0.1:8501...")
    env = os.environ.copy()
    env["BACKEND_URL"] = f"http://127.0.0.1:{port}"
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", "8501",
        "--server.address", "127.0.0.1"
    ], env=env)

    # Monitor processes
    try:
        while True:
            # Check if either process has exited
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
