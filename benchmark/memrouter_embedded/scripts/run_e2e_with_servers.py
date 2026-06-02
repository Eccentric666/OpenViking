#!/usr/bin/env python
"""Launch OV Server + Gateway with MemRouter enabled, then run E2E evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ov-root", type=Path, default=Path("D:/Code/cursorProject/OpenViking"))
    parser.add_argument("--echomem-root", type=Path, default=Path("D:/Code/cursorProject/EchoMem"))
    parser.add_argument("--ov-conf", type=Path, required=True)
    parser.add_argument("--ov-port", type=int, default=1934)
    parser.add_argument("--gw-port", type=int, default=1936)
    parser.add_argument("--bot-port", type=int, default=18790)
    parser.add_argument("--memrouter-config", type=Path, required=True)
    parser.add_argument("--route-events", type=Path, required=True)
    parser.add_argument("--eval-args", type=str, default="", help="Extra args for eval script")
    args = parser.parse_args()

    ov_root = str(args.ov_root)
    echomem_root = str(args.echomem_root)
    ov_conf = str(args.ov_conf)
    route_events = str(args.route_events)
    memrouter_config = str(args.memrouter_config)

    # Ensure route_events dir exists
    Path(route_events).parent.mkdir(parents=True, exist_ok=True)

    # Clear old route_events
    if Path(route_events).exists():
        Path(route_events).unlink()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"
    env["PYTHONPATH"] = f"{ov_root}/openviking/lib;{ov_root};{ov_root}/bot;{echomem_root}"
    env["OPENVIKING_CONFIG_FILE"] = ov_conf
    env["MEMROUTER_ENABLED"] = "true"
    env["MEMROUTER_CONFIG"] = memrouter_config
    env["ECHOMEM_PATH"] = echomem_root
    env["MEMROUTER_ROUTE_EVENTS"] = route_events

    procs = []

    try:
        # 1. Start OV Server
        print("[1/5] Starting OpenViking server...")
        ov_cmd = [
            sys.executable, "-X", "utf8",
            "-m", "uvicorn", "openviking.server.app:create_app",
            "--factory", "--host", "127.0.0.1", "--port", str(args.ov_port),
        ]
        ov_proc = subprocess.Popen(
            ov_cmd,
            cwd=ov_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(("ov_server", ov_proc))

        if not wait_for_port("127.0.0.1", args.ov_port, timeout=30):
            print("ERROR: OV Server did not start")
            sys.exit(1)
        print("      OV Server ready on port", args.ov_port)

        # 2. Start VikingBot gateway
        print("[2/5] Starting VikingBot gateway...")
        gw_cmd = [
            sys.executable, "-X", "utf8",
            "-m", "bot.vikingbot", "gateway",
            "--config", ov_conf,
            "--host", "127.0.0.1", "--port", str(args.bot_port),
        ]
        gw_proc = subprocess.Popen(
            gw_cmd,
            cwd=f"{ov_root}/bot",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(("gateway", gw_proc))

        if not wait_for_port("127.0.0.1", args.gw_port, timeout=30):
            print("ERROR: Gateway did not start")
            sys.exit(1)
        print("      Gateway ready on port", args.gw_port)

        # Wait for warm-up
        print("[3/5] Waiting 5s for warm-up...")
        time.sleep(5)

        # 3. Run evaluation
        print("[4/5] Running E2E evaluation...")
        eval_script = Path(__file__).parent / "eval_locomo_vikingbot_memrouter_e2e.py"
        eval_cmd = [
            sys.executable, "-X", "utf8",
            str(eval_script),
            "--dataset", f"{echomem_root}/benchmarks/locomo/memrouter_ov_e2e/data/locomo10.json",
            "--route-labels", f"{echomem_root}/benchmarks/locomo/memrouter_ov_e2e/data/locomo_e2e_route_labels.v3.jsonl",
            "--config", memrouter_config,
            "--ov-chat-endpoint", f"http://127.0.0.1:{args.gw_port}",
            "--ov-search-endpoint", f"http://127.0.0.1:{args.ov_port}",
            "--ov-api-key", "ov-test-key-12345",
            "--ov-account", "default",
            "--category", "3",
            "--route-events-path", route_events,
        ]
        if args.eval_args:
            eval_cmd.extend(args.eval_args.split())

        eval_env = env.copy()
        eval_proc = subprocess.run(eval_cmd, cwd=str(Path(__file__).parent.parent), env=eval_env)

        print("[5/5] Evaluation completed with exit code:", eval_proc.returncode)

    finally:
        print("Stopping servers...")
        for name, proc in procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            print(f"      {name} stopped")


if __name__ == "__main__":
    main()
