#!/usr/bin/env python
"""Full E2E: start services, run eval, stop services — all in one Python process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=1.0)
            sock.close()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    ov_root = Path("D:/Code/cursorProject/OpenViking")
    echomem_root = Path("D:/Code/cursorProject/EchoMem")
    config_dir = echomem_root / "benchmarks/locomo/memrouter_ov_e2e/configs"
    ov_conf = config_dir / "ov_1938.conf"
    memrouter_config = config_dir / "memrouter_eval.local.yaml"
    route_events = echomem_root / "benchmarks/locomo/memrouter_ov_e2e/logs/route_events_1938.jsonl"
    data_dir = echomem_root / "benchmarks/locomo/memrouter_ov_e2e/data"

    # Clear old route_events
    route_events.parent.mkdir(parents=True, exist_ok=True)
    if route_events.exists():
        route_events.unlink()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"
    env["PYTHONPATH"] = "D:/Code/cursorProject/OpenViking/openviking/lib;D:/Code/cursorProject/OpenViking;D:/Code/cursorProject/OpenViking/bot;D:/Code/cursorProject/EchoMem"
    env["OPENVIKING_CONFIG_FILE"] = str(ov_conf)
    env["MEMROUTER_ENABLED"] = "true"
    env["MEMROUTER_CONFIG"] = str(memrouter_config)
    env["ECHOMEM_PATH"] = str(echomem_root)
    env["MEMROUTER_ROUTE_EVENTS"] = str(route_events)

    procs: list[tuple[str, subprocess.Popen]] = []

    try:
        # 1. Start OV Server
        print("[1/5] Starting OpenViking server on port 1937...")
        ov_cmd = [
            sys.executable, "-X", "utf8",
            "-m", "uvicorn", "openviking.server.app:create_app",
            "--factory", "--host", "127.0.0.1", "--port", "1937",
        ]
        ov_proc = subprocess.Popen(
            ov_cmd,
            cwd=str(ov_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(("ov_server", ov_proc))

        if not wait_for_port("127.0.0.1", 1937, timeout=30):
            print("ERROR: OV Server did not start")
            return 1
        print("      OV Server ready")

        # 2. Start Gateway
        print("[2/5] Starting VikingBot gateway on port 18791...")
        gw_cmd = [
            sys.executable, "-X", "utf8",
            "-m", "vikingbot", "gateway",
            "--config", str(ov_conf),
            "--host", "127.0.0.1", "--port", "18791",
        ]
        gw_proc = subprocess.Popen(
            gw_cmd,
            cwd=str(ov_root / "bot"),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(("gateway", gw_proc))

        # Gateway exposes on 18791, but OV Server proxies from 1938
        # Check 1938 (OV Server with_bot proxy)
        print("[3/5] Waiting for gateway proxy on port 1938...")
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                import urllib.request
                req = urllib.request.Request(
                    "http://127.0.0.1:1938/bot/v1/chat",
                    data=json.dumps({"message": "hi", "session_id": "test"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    resp.read()
                    print("      Gateway proxy ready")
                    break
            except Exception:
                time.sleep(0.5)
        else:
            print("WARNING: Gateway proxy did not respond, proceeding anyway...")

        # Warm-up
        print("[4/5] Warm-up (5s)...")
        time.sleep(5)

        # 3. Run evaluation
        print("[5/5] Running E2E evaluation...")
        eval_script = Path(__file__).parent / "eval_locomo_vikingbot_memrouter_e2e.py"
        eval_cmd = [
            sys.executable, "-X", "utf8",
            str(eval_script),
            "--dataset", str(data_dir / "locomo10.json"),
            "--route-labels", str(data_dir / "locomo_e2e_route_labels.v3.jsonl"),
            "--memrouter-config", str(memrouter_config),
            "--ov-chat-endpoint", "http://127.0.0.1:1937",
            "--ov-search-endpoint", "http://127.0.0.1:1937",
            "--ov-api-key", "ov-test-key-12345",
            "--ov-account", "default",
            "--category", "3",
            "--route-events-path", str(route_events),
        ]

        eval_proc = subprocess.run(
            eval_cmd,
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )
        print(f"Evaluation exit code: {eval_proc.returncode}")
        return eval_proc.returncode

    finally:
        print("\nStopping services...")
        for name, proc in procs:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            except Exception as e:
                print(f"  Error stopping {name}: {e}")
            else:
                print(f"  {name} stopped")


if __name__ == "__main__":
    sys.exit(main())
