import json
import subprocess
import sys
import time


def test_gpu_job_array_runs_commands_with_round_robin_cuda_visible_devices(tmp_path):
    commands = tmp_path / "commands.jsonl"
    out0 = tmp_path / "gpu0.txt"
    out1 = tmp_path / "gpu1.txt"
    commands.write_text(
        "\n".join(
            [
                json.dumps({"name": "job0", "cmd": f"{sys.executable} -c \"import os; open('{out0}','w').write(os.environ.get('CUDA_VISIBLE_DEVICES',''))\""}),
                json.dumps({"name": "job1", "cmd": f"{sys.executable} -c \"import os; open('{out1}','w').write(os.environ.get('CUDA_VISIBLE_DEVICES',''))\""}),
            ]
        )
        + "\n"
    )
    summary = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.run_gpu_job_array",
            "--commands",
            str(commands),
            "--gpus",
            "0,1",
            "--summary",
            str(summary),
        ],
        check=True,
    )

    assert out0.read_text() == "0"
    assert out1.read_text() == "1"
    payload = json.loads(summary.read_text())
    assert [job["gpu"] for job in payload["jobs"]] == ["0", "1"]
    assert all(job["returncode"] == 0 for job in payload["jobs"])


def test_gpu_job_array_returns_failure_for_failed_command(tmp_path):
    commands = tmp_path / "commands.jsonl"
    commands.write_text(json.dumps({"name": "bad", "cmd": f"{sys.executable} -c \"raise SystemExit(7)\""}) + "\n")
    summary = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.run_gpu_job_array",
            "--commands",
            str(commands),
            "--gpus",
            "0",
            "--summary",
            str(summary),
        ]
    )

    assert result.returncode == 1
    payload = json.loads(summary.read_text())
    assert payload["jobs"][0]["returncode"] == 7


def test_gpu_job_array_allows_repeated_gpu_slots_for_concurrent_jobs(tmp_path):
    commands = tmp_path / "commands.jsonl"
    commands.write_text(
        "\n".join(
            [
                json.dumps({"name": "job0", "cmd": f"{sys.executable} -c \"import os,time; time.sleep(0.5); print(os.environ.get('CUDA_VISIBLE_DEVICES',''))\""}),
                json.dumps({"name": "job1", "cmd": f"{sys.executable} -c \"import os,time; time.sleep(0.5); print(os.environ.get('CUDA_VISIBLE_DEVICES',''))\""}),
            ]
        )
        + "\n"
    )
    summary = tmp_path / "summary.json"

    started_at = time.time()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.run_gpu_job_array",
            "--commands",
            str(commands),
            "--gpus",
            "0,0",
            "--summary",
            str(summary),
        ],
        check=True,
    )
    elapsed = time.time() - started_at

    payload = json.loads(summary.read_text())
    assert [job["gpu"] for job in payload["jobs"]] == ["0", "0"]
    assert elapsed < 0.9
