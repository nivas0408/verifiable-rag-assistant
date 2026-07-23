import subprocess
import sys
import os
import time

def main():
    # 1. Start FastAPI in the background on localhost:8000
    print("Starting FastAPI backend internally on 127.0.0.1:8000...")
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", "8000"
    ])

    # Give Uvicorn a couple of seconds to bind to port 8000
    time.sleep(3)

    # 2. Start Streamlit on the public port (supplied by Render as $PORT, default 8501)
    port = os.getenv("PORT", "8501")
    print(f"Starting Streamlit frontend on 0.0.0.0:{port}...")
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", port, "--server.address", "0.0.0.0"
    ])

    # 3. Monitor processes
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
