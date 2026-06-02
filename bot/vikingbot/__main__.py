"""
Entry point for running vikingbot as a module: python -m vikingbot
"""

import io
import os
import sys

# UTF-8 guard (redundant with vikingbot/__init__.py but safe as a fallback)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    # Force stdout/stderr to UTF-8 to avoid gbk codec errors on Windows
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from vikingbot.cli.commands import app

if __name__ == "__main__":
    # sys.argv = sys.argv + ['gateway']
    app()
