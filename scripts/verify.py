#!/usr/bin/env python
"""One command to check that every phase works on YOUR machine.

    python scripts/verify.py                      # offline, uses a tiny random model (~1-2 min)
    python scripts/verify.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0   # + real-model benchmarks
    python scripts/verify.py --skip-server        # if you can't open a local port

Runs, in order: environment report, the full test-suite, then a smoke run of each phase's
benchmark / server. Prints a PASS/FAIL table, writes full logs to results/verify.log, and exits
non-zero if anything failed. Tests that need a GPU or Triton are reported as SKIPPED, not failed.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = ROOT / "results" / "verify.log"


def run(cmd, log, timeout=1800, env=None):
    log.write(f"\n$ {' '.join(map(str, cmd))}\n")
    log.flush()
    p = subprocess.run(
        [str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env
    )
    log.write(p.stdout + p.stderr)
    log.flush()
    return p.returncode == 0, (p.stdout + p.stderr)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny", help="'tiny' (offline) or an HF repo id")
    ap.add_argument("--skip-server", action="store_true")
    args = ap.parse_args()
    real = args.model != "tiny"
    LOG.parent.mkdir(exist_ok=True)
    results = []

    with open(LOG, "w") as log:
        def step(name, fn):
            t = time.time()
            try:
                ok, note = fn()
            except Exception as e:  # noqa: BLE001
                ok, note = False, f"{type(e).__name__}: {e}"
            results.append((name, "PASS" if ok else "FAIL", time.time() - t, note))
            print(f"[{'PASS' if ok else 'FAIL'}] {name} ({time.time() - t:.1f}s)", flush=True)

        def env():
            code = ("import torch,platform;print('python',platform.python_version(),'| torch',torch.__version__);"
                    "print('cuda',torch.cuda.is_available(),"
                    "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '');"
                    "\ntry:\n import triton;print('triton',triton.__version__)\n"
                    "except Exception as e:\n print('triton unavailable (INT4 uses the PyTorch path)')")
            ok, out = run([PY, "-c", code], log)
            print("      " + out.strip().replace("\n", "\n      "))
            return ok, ""

        def tests():
            cpu_env = os.environ.copy()
            cpu_env["TRITON_INTERPRET"] = "1"
            cpu_ok, cpu_out = run(
                [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 "-k", "not test_kernel_matches_reference_on_gpu"],
                log, env=cpu_env,
            )
            cpu_summary = [line for line in cpu_out.splitlines() if " passed" in line or " failed" in line]
            cpu_note = cpu_summary[-1] if cpu_summary else cpu_out.strip().splitlines()[-1]

            gpu_env = os.environ.copy()
            gpu_env.pop("TRITON_INTERPRET", None)
            available, capability = run(
                [PY, "-c",
                 "import importlib.util, torch; "
                 "print(torch.cuda.is_available() and importlib.util.find_spec('triton') is not None)"],
                log,
            )
            if not available:
                return False, f"{cpu_note}; could not check GPU Triton availability"
            if capability.strip().splitlines()[-1] == "True":
                gpu_ok, gpu_out = run(
                    [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                     "tests/test_triton_int4.py", "-k", "gpu"],
                    log, env=gpu_env,
                )
                gpu_summary = [line for line in gpu_out.splitlines() if " passed" in line or " failed" in line]
                gpu_note = gpu_summary[-1] if gpu_summary else gpu_out.strip().splitlines()[-1]
            else:
                gpu_ok, gpu_note = True, "GPU Triton tests skipped (CUDA GPU or Triton unavailable)"

            note = f"{cpu_note}; {gpu_note}"
            print("      " + note)
            return cpu_ok and gpu_ok, note

        m = args.model
        step("environment", env)
        step("tests: all phases (correctness vs HF, paged cache, scheduler, quant, kernel, server)", tests)
        step("Phase 1  naive vs KV-cache benchmark",
             lambda: run([PY, "benchmarks/run_benchmark.py", "--model", m, "--prompt-lens", "32" if not real else "128",
                          "--new-tokens", "24" if not real else "64", "--runs", "2"], log))
        step("Phase 2  paged vs contiguous capacity",
             lambda: run([PY, "benchmarks/memory_capacity.py", "--preset", "tinyllama-1.1b", "--trials", "5"], log))
        step("Phase 3  continuous vs static vs sequential load test",
             lambda: run([PY, "benchmarks/load_test.py", "--model", m, "--rates", "20" if not real else "2",
                          "--num-requests", "12" if not real else "24", "--prompt-range", "8", "24",
                          "--out-range", "8", "16"], log))
        step("Phase 4  fp16 / int8 / int4 benchmark",
             lambda: run([PY, "benchmarks/run_quant_benchmark.py", "--model", m, "--runs", "2",
                          "--new-tokens", "24" if not real else "64"], log))

        def server():
            port = free_port()
            proc = subprocess.Popen([PY, "-m", "llmserve.server", "--model", m, "--port", str(port)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                url = f"http://127.0.0.1:{port}"
                for _ in range(240):
                    try:
                        urllib.request.urlopen(url + "/health", timeout=1)
                        break
                    except Exception:  # noqa: BLE001
                        if proc.poll() is not None:
                            return False, "server exited early (see results/verify.log)"
                        time.sleep(0.5)
                else:
                    return False, "server did not become healthy"
                vocab = "256" if m == "tiny" else "32000"
                return run([PY, "benchmarks/load_test.py", "--url", url, "--vocab", vocab, "--rates", "10",
                            "--num-requests", "10", "--prompt-range", "8", "16", "--out-range", "6", "10"], log)
            finally:
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()

        if args.skip_server:
            results.append(("Phase 5  OpenAI-compatible server + HTTP load test", "SKIP", 0.0, ""))
        else:
            step("Phase 5  OpenAI-compatible server + HTTP load test", server)
        step("Phase 6  profiler (decode steps -> Chrome trace)",
             lambda: run([PY, "benchmarks/profile_decode.py", "--model", m, "--steps", "5"], log))
        step("Phase 6  INT4 Triton kernel micro-benchmark (GPU only; skips cleanly otherwise)",
             lambda: run([PY, "benchmarks/bench_int4_kernel.py"], log))

    print("\n" + "=" * 78)
    for name, status, secs, _ in results:
        print(f"{status:5} {secs:6.1f}s  {name}")
    failed = [r for r in results if r[1] == "FAIL"]
    print("=" * 78)
    print("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED - see {LOG}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
