import subprocess
import sys
import time
import webbrowser
import os
import signal
import atexit

_processes = []


def cleanup():
    for proc in _processes:
        if proc.poll() is None:
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True
                    )
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def signal_handler(signum, frame):
    print("\n\nShutting down...")
    cleanup()
    sys.exit(0)


def main():
    print("\nStarting Quant Analytics...\n")
    
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)
    
    root = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root, "backend")
    frontend_dir = os.path.join(root, "frontend")
    
    print("Starting backend...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"],
        cwd=backend_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _processes.append(backend)
    
    time.sleep(3)
    
    print("Starting frontend...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"],
        cwd=frontend_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _processes.append(frontend)
    
    print("\nBackend:  http://localhost:8000/docs")
    print("Frontend: http://localhost:8501")
    print("\nPress Ctrl+C to stop\n")
    
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass
    
    try:
        while True:
            if backend.poll() is not None:
                print("Backend stopped unexpectedly")
                break
            if frontend.poll() is not None:
                print("Frontend stopped unexpectedly")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
