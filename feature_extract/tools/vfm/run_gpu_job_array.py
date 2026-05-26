"""Run independent VFM experiment commands across GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class JobSpec:
    name: str
    cmd: str


@dataclass(frozen=True)
class JobResult:
    name: str
    cmd: str
    gpu: str
    returncode: int
    elapsed_s: float


def _load_jobs(path: Path) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    for idx, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        jobs.append(JobSpec(name=str(item.get("name", f"job{idx}")), cmd=str(item["cmd"])))
    if not jobs:
        raise ValueError("commands file must contain at least one job")
    return jobs


def _parse_gpus(text: str) -> list[str]:
    gpus = [item.strip() for item in text.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def run_job_array(jobs: list[JobSpec], gpus: list[str]) -> list[JobResult]:
    pending = list(enumerate(jobs))
    running: dict[int, tuple[str, int, JobSpec, subprocess.Popen, float]] = {}
    results: list[JobResult] = []

    while pending or running:
        for slot_idx, gpu in enumerate(gpus):
            if slot_idx in running or not pending:
                continue
            _job_idx, job = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(job.cmd, shell=True, env=env)
            running[slot_idx] = (gpu, _job_idx, job, process, time.time())

        time.sleep(0.05)
        for slot_idx, (gpu, _job_idx, job, process, started_at) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            results.append(
                JobResult(
                    name=job.name,
                    cmd=job.cmd,
                    gpu=gpu,
                    returncode=int(returncode),
                    elapsed_s=float(time.time() - started_at),
                )
            )
            del running[slot_idx]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent commands across CUDA_VISIBLE_DEVICES slots")
    parser.add_argument("--commands", required=True, help="JSONL with fields: name, cmd")
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids, e.g. 0,1")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    jobs = _load_jobs(Path(args.commands))
    results = run_job_array(jobs, _parse_gpus(args.gpus))
    payload = {
        "job_count": len(results),
        "failed_count": sum(1 for result in results if result.returncode != 0),
        "jobs": [asdict(result) for result in results],
    }
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["failed_count"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
