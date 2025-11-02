#!/usr/bin/env python3
"""
GPU stress test using PyTorch.

Usage (defaults are safe-ish; adjust carefully):
    python gpu_stress_test.py --tensor-size 4096 --iterations 1000 --duration 300 --temp-limit 85 --ramp-up 10

Defaults:
    tensor_size=4096
    iterations=1000
    duration_seconds=300
    temp_limit_c=85
    ramp_up_seconds=10

Safety notes:
 - Monitor temps with external tools; stop if temperatures climb unexpectedly.
 - Do NOT run unattended on hardware you cannot monitor.
 - Use smaller tensor_size or fewer iterations if you see instability.
"""

import argparse
import time
import subprocess
import sys
import signal
import torch

def get_nvidia_smi():
    """Return output of nvidia-smi or None if not available."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None

def parse_smi_line(line):
    """Parse one-line CSV: temp,util,mem_used,mem_total -> ints"""
    try:
        parts = [p.strip() for p in line.split(",")]
        return {
            "temp": int(parts[0]),
            "util": int(parts[1].replace("%","")),
            "mem_used": int(parts[2]),
            "mem_total": int(parts[3])
        }
    except Exception:
        return None

stop_requested = False
def signal_handler(sig, frame):
    global stop_requested
    stop_requested = True
    print("\nKeyboard interrupt received. Stopping...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    parser = argparse.ArgumentParser(description="PyTorch GPU stress test with safety monitoring.")
    parser.add_argument("--tensor-size", type=int, default=4096, help="Square matrix size (default: 4096)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of matmul iterations (default: 1000)")
    parser.add_argument("--duration", type=int, default=300, help="Max duration in seconds (default: 300)")
    parser.add_argument("--temp-limit", type=int, default=85, help="GPU temperature limit in C to stop (default: 85)")
    parser.add_argument("--ramp-up", type=int, default=10, help="Ramp-up seconds to gradually increase load (default: 10)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)

    device = torch.device("cuda:0")
    print("Using device:", torch.cuda.get_device_name(0))
    print(f"Tensor size: {args.tensor_size}x{args.tensor_size}, iterations: {args.iterations}, duration: {args.duration}s")
    print(f"Temp limit: {args.temp_limit}C, ramp up: {args.ramp_up}s")

    # Pre-allocate a matrix to reuse (reduces host->device noise)
    try:
        a = torch.randn(args.tensor_size, args.tensor_size, device=device)
        torch.cuda.synchronize()
    except Exception as e:
        print("Failed to allocate tensors on GPU:", e)
        sys.exit(1)

    start_time = time.time()
    last_report = start_time
    iteration = 0

    try:
        # Optional ramp-up loop: perform small ops then increase
        ramp_start = time.time()
        while time.time() - ramp_start < args.ramp_up and not stop_requested:
            b = torch.randn(args.tensor_size//4, args.tensor_size//4, device=device)
            c = torch.matmul(b, b)
            torch.cuda.synchronize()
            del b, c
        torch.cuda.empty_cache()

        # Main stress loop
        while iteration < args.iterations and (time.time() - start_time) < args.duration and not stop_requested:
            # Build a new random tensor for each iter to maximize work
            b = torch.randn(args.tensor_size, args.tensor_size, device=device)
            t0 = time.time()
            c = torch.matmul(a, b)
            # ensure ops complete
            torch.cuda.synchronize()
            t1 = time.time()

            # periodic reporting
            now = time.time()
            if now - last_report >= 1.0:
                smi = get_nvidia_smi()
                if smi:
                    parsed = parse_smi_line(smi.splitlines()[0])
                    if parsed:
                        print(f"[{iteration}/{args.iterations}] elapsed={now-start_time:.1f}s matmul_time={(t1-t0):.3f}s temp={parsed['temp']}C util={parsed['util']}% mem={parsed['mem_used']}/{parsed['mem_total']} MiB")
                        if parsed['temp'] >= args.temp_limit:
                            print(f"Temperature {parsed['temp']}C >= limit {args.temp_limit}C. Stopping for safety.")
                            break
                else:
                    # fallback reporting if nvidia-smi not available
                    allocated = torch.cuda.memory_allocated(device)
                    print(f"[{iteration}/{args.iterations}] elapsed={now-start_time:.1f}s matmul_time={(t1-t0):.3f}s allocated={allocated/1e6:.1f} MB")

                last_report = now

            # cleanup per iteration
            del b, c
            torch.cuda.empty_cache()

            iteration += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print("Error during stress loop:", e)
    finally:
        # final status
        total_elapsed = time.time() - start_time
        print(f"Finished. Iterations completed: {iteration}. Total time: {total_elapsed:.1f}s")
        # cleanup large tensor
        try:
            del a
        except Exception:
            pass
        torch.cuda.empty_cache()
        # final nvidia-smi snapshot
        smi = get_nvidia_smi()
        if smi:
            parsed = parse_smi_line(smi.splitlines()[0])
            if parsed:
                print(f"Final GPU snapshot: temp={parsed['temp']}C util={parsed['util']}% mem={parsed['mem_used']}/{parsed['mem_total']} MiB")

if __name__ == "__main__":
    main()

