import os
import subprocess
import sys

import pytest


@pytest.mark.smoke
def test_cli_process_exits_cleanly(tmp_path):
    """CLIが起動し、終了コマンドで正常終了するかを確認するスモークテスト。"""
    env = os.environ.copy()
    env.update(
        {
            "CHAT_HISTORY_USER_ID": "smoke-user",
            "GOOGLE_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key",
            "PYTHONUNBUFFERED": "1",
        }
    )

    proc = subprocess.run(
        [sys.executable, "chat_logic.py"],
        input="quit\n",
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stdout
