"""Start one local backend and one UI; Ctrl-C stops both child processes."""

import signal
import subprocess
import sys

from patent_agent.settings import ROOT, Settings

s = Settings()
children = []


def stop(*_):
    for child in children:
        child.terminate()
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
    raise SystemExit(0)


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
try:
    children.append(
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "patent_agent.api:app",
                "--host",
                s.api_host,
                "--port",
                str(s.api_port),
                "--workers",
                "1",
            ],
            cwd=ROOT,
        )
    )
    children.append(
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "src/patent_agent/ui.py",
                "--server.address",
                s.ui_host,
                "--server.port",
                str(s.ui_port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=ROOT,
        )
    )
    print(f"页面：http://{s.ui_host}:{s.ui_port}  API：http://{s.api_host}:{s.api_port}", flush=True)
    while all(c.poll() is None for c in children):
        import time

        time.sleep(1)
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
