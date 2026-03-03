from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorboardX import SummaryWriter

# Allow running as: python scripts/optimize.py
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.base import RunConfig
from config.registry import available_tasks, get_task_config


def _make_run_dir(task_name: str, runs_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_root) / task_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_task(cfg: RunConfig, run_dir: Path, logger: Any, task_name: str) -> dict[str, Any]:
    return cfg.runner(
        cfg=cfg,
        run_dir=run_dir,
        logger=logger,
        task_name=task_name,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified optimization entrypoint.")
    parser.add_argument("--task", type=str, help="Task key, e.g. optimal_torque_roki2.single_motor_baseline")
    parser.add_argument("--list-tasks", action="store_true", help="List registered task keys and exit.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.list_tasks:
        for task_key in available_tasks():
            print(task_key)
        return 0

    if not args.task:
        raise ValueError("Missing required --task (or use --list-tasks).")

    cfg = get_task_config(args.task)
    runs_root = cfg.logging.runs_root
    run_dir = _make_run_dir(args.task, runs_root)

    logger = SummaryWriter(log_dir=str(run_dir))
    try:
        resolved_cfg = cfg.to_dict()
        (run_dir / "config.json").write_text(json.dumps(resolved_cfg, indent=2), encoding="utf-8")
        logger.add_text("config/resolved", json.dumps(resolved_cfg, indent=2), 0)

        result = _run_task(cfg=cfg, run_dir=run_dir, logger=logger, task_name=args.task)
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.add_text("run/result", json.dumps(result, indent=2), 0)
        logger.flush()

        print(f"Run directory: {run_dir}")
        print(json.dumps(result, indent=2))
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
