# -*- coding: utf-8 -*-
"""Sequential experiment queue driver (multi-queue capable).

Usage: python run_queue.py [jobs_file]   (default: jobs.json)

Reads the given jobs JSON file: list of dicts with keys forwarded to
run_exp.py (model, data, name, seed, cb, pretrained, ...). Skips jobs whose
`name` already appears in runs/wafer/results.jsonl. Lock file is per
jobs-file so multiple disjoint queues can run in parallel on one machine.
Designed to be re-runnable.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'runs', 'wafer', 'results.jsonl')


def done_names():
    if not os.path.exists(RESULTS):
        return set()
    out = set()
    for line in open(RESULTS, encoding='utf-8'):
        line = line.strip()
        if line:
            try:
                out.add(json.loads(line)['name'])
            except Exception:
                pass
    return out


def main():
    jobs_file = sys.argv[1] if len(sys.argv) > 1 else 'jobs.json'
    jobs_path = jobs_file if os.path.isabs(jobs_file) else os.path.join(HERE, jobs_file)
    tag = os.path.splitext(os.path.basename(jobs_path))[0]
    lock = os.path.join(HERE, 'runs', 'wafer', f'queue_{tag}.lock')
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print('[abort] another queue instance holds', lock)
        return
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        jobs = json.load(open(jobs_path, encoding='utf-8'))
        done = done_names()
        print(f'[queue] jobs_file={os.path.basename(jobs_path)} jobs={len(jobs)} done={len(done)}', flush=True)
        for j in jobs:
            if j['name'] in done:
                print(f"[skip] {j['name']}")
                continue
            if 'data' in j and not os.path.exists(str(j['data'])):
                print(f"[wait] {j['name']}: data yaml missing: {j['data']}")
                continue
            missing = False
            for key in ('model', 'pretrained'):
                if j.get(key) and str(j[key]).endswith(('.pt', '.yaml')) and not os.path.exists(str(j[key])):
                    print(f"[wait] {j['name']}: file missing: {j[key]}")
                    missing = True
                    break
            if missing:
                continue
            cmd = [sys.executable, os.path.join(HERE, 'run_exp.py')]
            for k, v in j.items():
                if k == 'cb' and v:
                    cmd.append('--cb')
                elif k != 'cb':
                    cmd += [f'--{k}', str(v)]
            print(f"[run ] {j['name']}: {' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd, cwd=HERE)
            if r.returncode != 0:
                print(f"[FAIL] {j['name']} rc={r.returncode}", flush=True)
                with open(os.path.join(HERE, 'runs', 'wafer', 'failures.log'), 'a', encoding='utf-8') as f:
                    f.write(f"{j['name']} rc={r.returncode}\n")
    finally:
        os.remove(lock)
    print(f'[queue] {tag}: all jobs processed')


if __name__ == '__main__':
    main()
