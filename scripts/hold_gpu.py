#!/usr/bin/env python3
"""
占用指定 GPU 显存，防止被抢占。

用法:
  锁定:   bash scripts/gpu_lock.sh on 3 4
  释放:   bash scripts/gpu_lock.sh off
  状态:   bash scripts/gpu_lock.sh status

也可直接运行:
  python scripts/hold_gpu.py 3 4          # 前台运行, Ctrl+C 释放
  python scripts/hold_gpu.py 3 4 --gb 20  # 每卡占 20GB
"""
import sys
import torch
import time
import signal
import os


def get_total_gb(device):
    props = torch.cuda.get_device_properties(device)
    for attr in ('total_mem', 'total_memory'):
        if hasattr(props, attr):
            return getattr(props, attr) / 1024**3
    return 24.0  # fallback for 4090


def main():
    import argparse
    parser = argparse.ArgumentParser(description='锁定 GPU 显存')
    parser.add_argument('gpus', nargs='*', type=int, default=[3, 4],
                        help='要锁定的 GPU ID (默认: 3 4)')
    parser.add_argument('--gb', type=float, default=22,
                        help='每卡占用显存 GB (默认: 22)')
    parser.add_argument('--util', type=int, default=15,
                        help='目标 GPU 利用率 %% (默认: 15, 范围 1-95)')
    args = parser.parse_args()

    # 写 PID 文件
    pid_file = os.path.join(os.path.dirname(__file__), '..', 'output', 'logs', 'hold_gpu.pid')
    os.makedirs(os.path.dirname(pid_file), exist_ok=True)
    with open(pid_file, 'w') as f:
        f.write(f"{os.getpid()}\n{','.join(map(str, args.gpus))}\n")

    tensors = []
    for gpu_id in args.gpus:
        dev = torch.device(f'cuda:{gpu_id}')
        total_gb = get_total_gb(dev)
        alloc_gb = min(args.gb, total_gb - 1)
        alloc_bytes = int(alloc_gb * 1024**3)
        name = torch.cuda.get_device_properties(dev).name
        print(f"GPU {gpu_id} ({name}, {total_gb:.0f}GB): 占用 {alloc_gb:.0f}GB...")
        t = torch.empty(alloc_bytes // 4, dtype=torch.float32, device=dev)
        tensors.append(t)
        used = torch.cuda.memory_allocated(dev) / 1024**3
        print(f"  ✓ 已占用 {used:.1f}GB")

    print(f"\n已锁定 GPU {args.gpus}，按 Ctrl+C 或 kill {os.getpid()} 释放。")

    def cleanup(*_):
        print("\n释放显存...")
        tensors.clear()
        torch.cuda.empty_cache()
        try:
            os.remove(pid_file)
        except OSError:
            pass
        print("已释放。")
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # 每张卡一个线程，并行计算，保持目标利用率
    target_util = max(1, min(95, args.util))
    print(f"目标利用率: ~{target_util}%")

    import threading

    def gpu_worker(gpu_id, target_util):
        dev = torch.device(f'cuda:{gpu_id}')
        mat_size = 4096 if target_util >= 50 else 2048
        a = torch.randn(mat_size, mat_size, device=dev)
        b = torch.randn(mat_size, mat_size, device=dev)
        batch_count = max(1, target_util // 5)
        sleep_time = max(0.005, (100 - target_util) / 100 * 1.5)

        while not stop_event.is_set():
            for _ in range(batch_count):
                c = torch.mm(a, b)
                c += a
            torch.cuda.synchronize(dev)
            if sleep_time > 0.01:
                time.sleep(sleep_time)

    stop_event = threading.Event()
    threads = []
    for gpu_id in args.gpus:
        t = threading.Thread(target=gpu_worker, args=(gpu_id, target_util), daemon=True)
        t.start()
        threads.append(t)
        print(f"  GPU {gpu_id} worker 已启动")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        cleanup()


if __name__ == '__main__':
    main()
