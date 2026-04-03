import json
import time
import subprocess
from pathlib import Path

RUN_NAMES = [
    "full_v4_2_seed42_fixed_nowarm_lambda1",
    "full_v4_2_seed42_fixed_warmup_lambda1",
    "wo_semantics_v4_2_seed42",
    "wo_object_bias_v4_2_seed42",
]
INTERVAL = 10
SAMPLES = 31  # ~= 5 minutes
REPORT = Path("reports/d1_resource_timeseries.json")
REPORT.parent.mkdir(parents=True, exist_ok=True)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, text=True, capture_output=True).stdout


def parse_ps():
    out = sh("ps -eo pid,ppid,pcpu,pmem,etimes,args")
    rows = []
    lines = out.strip().splitlines()
    for ln in lines[1:]:
        parts = ln.strip().split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, pcpu, pmem, etimes, args = parts
        try:
            rows.append(
                {
                    "pid": int(pid),
                    "ppid": int(ppid),
                    "pcpu": float(pcpu),
                    "pmem": float(pmem),
                    "etimes": int(float(etimes)),
                    "args": args,
                }
            )
        except Exception:
            continue
    return rows


def classify_runs(ps_rows):
    run_map = {}
    for rn in RUN_NAMES:
        cand = [
            r
            for r in ps_rows
            if f"--run-name {rn}" in r["args"] and "train_stwm_v4_2_real.py" in r["args"]
        ]
        wrapper = [r for r in cand if "conda run --no-capture-output -n stwm python" in r["args"]]
        main = [r for r in cand if r["args"].startswith("python ")]

        main_pid = None
        if main:
            scored = []
            for r in main:
                child_count = sum(1 for x in cand if x["ppid"] == r["pid"])
                scored.append((child_count, r["pcpu"], r["pid"]))
            scored.sort(reverse=True)
            main_pid = scored[0][2]

        wrapper_pid = wrapper[0]["pid"] if wrapper else None
        worker_pids = [r["pid"] for r in cand if main_pid is not None and r["ppid"] == main_pid]

        etimes = 0
        pcpu = 0.0
        pmem = 0.0
        if main_pid is not None:
            mr = next((r for r in main if r["pid"] == main_pid), None)
            if mr:
                etimes = mr["etimes"]
                pcpu = mr["pcpu"]
                pmem = mr["pmem"]

        run_map[rn] = {
            "wrapper_pid": wrapper_pid,
            "train_pid": main_pid,
            "worker_pids": worker_pids,
            "active": main_pid is not None,
            "train_pcpu": pcpu,
            "train_pmem": pmem,
            "train_etimes": etimes,
        }
    return run_map


def read_proc_stat():
    total = None
    per_core = {}
    with open("/proc/stat", "r") as f:
        for ln in f:
            if not ln.startswith("cpu"):
                break
            parts = ln.split()
            name = parts[0]
            vals = list(map(int, parts[1:]))
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total_t = sum(vals)
            if name == "cpu":
                total = (idle, total_t)
            else:
                per_core[name] = (idle, total_t)
    return total, per_core


def cpu_usage(prev, cur):
    pi, pt = prev
    ci, ct = cur
    dt = ct - pt
    di = ci - pi
    if dt <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (dt - di) / dt))


def mem_info():
    kv = {}
    with open("/proc/meminfo", "r") as f:
        for ln in f:
            k, v = ln.split(":", 1)
            kv[k.strip()] = int(v.strip().split()[0])
    total = kv.get("MemTotal", 0) * 1024
    avail = kv.get("MemAvailable", 0) * 1024
    used = max(0, total - avail)
    return {"mem_total": total, "mem_used": used, "mem_available": avail}


def disk_device_for_path(path):
    src = sh(f"df --output=source {path} | tail -n 1").strip()
    return src.split("/")[-1] if src.startswith("/dev/") else src


def read_diskstats(dev):
    with open("/proc/diskstats", "r") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 14:
                continue
            if parts[2] == dev:
                sectors_read = int(parts[5])
                sectors_written = int(parts[9])
                return sectors_read, sectors_written
    return 0, 0


def nvidia_gpus():
    q = "index,utilization.gpu,memory.used,memory.total,pstate,utilization.encoder,utilization.decoder"
    out = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits")
    rows = []
    for ln in out.strip().splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 7:
            continue
        try:
            rows.append(
                {
                    "index": int(float(p[0])),
                    "util_gpu": float(p[1]),
                    "mem_used_mib": float(p[2]),
                    "mem_total_mib": float(p[3]),
                    "pstate": p[4],
                    "util_encoder": float(p[5]),
                    "util_decoder": float(p[6]),
                }
            )
        except Exception:
            continue
    return rows


def nvidia_compute_apps():
    out = sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits")
    rows = []
    for ln in out.strip().splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 3:
            continue
        try:
            rows.append({"gpu_uuid": p[0], "pid": int(float(p[1])), "used_mib": float(p[2])})
        except Exception:
            continue
    return rows


def gpu_uuid_map():
    out = sh("nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits")
    m = {}
    for ln in out.strip().splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 2:
            continue
        try:
            m[p[1]] = int(float(p[0]))
        except Exception:
            pass
    return m


def read_pid_io(pid):
    try:
        rb = wb = 0
        with open(f"/proc/{pid}/io", "r") as f:
            for ln in f:
                if ln.startswith("read_bytes:"):
                    rb = int(ln.split(":", 1)[1].strip())
                elif ln.startswith("write_bytes:"):
                    wb = int(ln.split(":", 1)[1].strip())
        return rb, wb
    except Exception:
        return None


def loadavg():
    with open("/proc/loadavg", "r") as f:
        p = f.read().strip().split()
    return {"load1": float(p[0]), "load5": float(p[1]), "load15": float(p[2])}


def main():
    dev = disk_device_for_path("/home/chen034/workspace/stwm")
    gpu_uuid_to_idx = gpu_uuid_map()

    samples = []
    prev_total, prev_cores = read_proc_stat()
    prev_disk = read_diskstats(dev)
    prev_pid_io = {}
    prev_ts = time.time()

    for i in range(SAMPLES):
        ts = time.time()
        ps_rows = parse_ps()
        runs = classify_runs(ps_rows)
        gpus = nvidia_gpus()
        apps = nvidia_compute_apps()
        mem = mem_info()
        la = loadavg()
        cur_total, cur_cores = read_proc_stat()
        cur_disk = read_diskstats(dev)

        dt = max(1e-6, ts - prev_ts)
        cpu_total_pct = cpu_usage(prev_total, cur_total)
        core_pcts = {k: cpu_usage(prev_cores.get(k, cur_cores[k]), v) for k, v in cur_cores.items()}
        top_cores = sorted(core_pcts.items(), key=lambda x: x[1], reverse=True)[:8]

        dr = max(0, cur_disk[0] - prev_disk[0]) * 512.0 / (1024.0 * 1024.0) / dt
        dw = max(0, cur_disk[1] - prev_disk[1]) * 512.0 / (1024.0 * 1024.0) / dt

        pid_io_delta = {}
        pids_to_track = set()
        for rv in runs.values():
            if rv["wrapper_pid"]:
                pids_to_track.add(rv["wrapper_pid"])
            if rv["train_pid"]:
                pids_to_track.add(rv["train_pid"])
            for p in rv["worker_pids"]:
                pids_to_track.add(p)

        for pid in pids_to_track:
            cur = read_pid_io(pid)
            if cur is None:
                continue
            prev = prev_pid_io.get(pid)
            if prev is not None:
                rb = max(0, cur[0] - prev[0])
                wb = max(0, cur[1] - prev[1])
                pid_io_delta[pid] = {
                    "read_MBps": rb / (1024.0 * 1024.0) / dt,
                    "write_MBps": wb / (1024.0 * 1024.0) / dt,
                }
            prev_pid_io[pid] = cur

        run_stats = {}
        for rn, rv in runs.items():
            pset = set([p for p in [rv.get("wrapper_pid"), rv.get("train_pid")] if p]) | set(rv.get("worker_pids", []))
            r_mb = sum(pid_io_delta.get(p, {}).get("read_MBps", 0.0) for p in pset)
            w_mb = sum(pid_io_delta.get(p, {}).get("write_MBps", 0.0) for p in pset)

            gpu_idx = None
            if rv.get("train_pid"):
                for a in apps:
                    if a["pid"] == rv["train_pid"]:
                        gpu_idx = gpu_uuid_to_idx.get(a["gpu_uuid"])
                        break

            run_stats[rn] = {
                **rv,
                "gpu_idx_live": gpu_idx,
                "io_read_MBps": r_mb,
                "io_write_MBps": w_mb,
            }

        top_io = sorted(pid_io_delta.items(), key=lambda kv: kv[1]["read_MBps"], reverse=True)[:12]
        top_io_fmt = [{"pid": int(pid), **vals} for pid, vals in top_io]

        sample = {
            "ts": ts,
            "iso_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
            "interval_s": dt,
            "loadavg": la,
            "cpu_total_pct": cpu_total_pct,
            "cpu_top_cores": [{"core": k, "pct": v} for k, v in top_cores],
            "memory": mem,
            "disk_device": dev,
            "disk_read_MBps": dr,
            "disk_write_MBps": dw,
            "gpus": gpus,
            "runs": run_stats,
            "top_pid_io_read": top_io_fmt,
        }
        samples.append(sample)

        print(
            f"[sample {i + 1}/{SAMPLES}] {sample['iso_time']} cpu={cpu_total_pct:.1f}% diskR={dr:.1f}MB/s diskW={dw:.1f}MB/s",
            flush=True,
        )

        prev_total, prev_cores = cur_total, cur_cores
        prev_disk = cur_disk
        prev_ts = ts
        if i < SAMPLES - 1:
            time.sleep(INTERVAL)

    REPORT.write_text(json.dumps({"interval_s": INTERVAL, "samples": samples}, indent=2, ensure_ascii=False))
    print(f"saved={REPORT}", flush=True)


if __name__ == "__main__":
    main()
