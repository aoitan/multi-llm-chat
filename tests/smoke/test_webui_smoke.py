import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest


def _wait_for_http(url: str, timeout: float = 20.0, interval: float = 0.5) -> bool:
    """指定URLが200を返すまでポーリングする。"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError):
            time.sleep(interval)
            continue
    return False


@pytest.mark.smoke
def test_webui_process_starts_and_serves_index():
    """WebUIが起動しHTTPで応答するかを確認するスモークテスト。"""
    env = os.environ.copy()
    env.update(
        {
            "GRADIO_SERVER_NAME": "127.0.0.1",
            "GRADIO_SERVER_PORT": "7865",
            "MLC_SERVER_NAME": "127.0.0.1",
            "GOOGLE_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key",
        }
    )

    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        assert _wait_for_http("http://127.0.0.1:7865/"), "WebUI did not start in time"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # terminate() で終了させるため SIGTERM (-15) を許容
    assert proc.returncode in (0, None, -15)
