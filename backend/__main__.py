"""Start the app as a single process: UI + API on one port."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "frontend" / "dist" / "index.html"


def main() -> None:
    if not DIST.exists():
        frontend = ROOT / "frontend"
        print("Building the UI into frontend/dist ...")
        subprocess.run(["npm", "run", "build"], cwd=frontend, check=True)

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Open http://{host}:{port}")
    import uvicorn

    uvicorn.run("backend.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
