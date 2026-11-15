"""Reproducible Windows revision benchmarks; no synthetic performance workloads."""
from __future__ import annotations

import csv
import datetime
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import h5py
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KEY = 'Acquisition/Raw[0]/RawData'


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def metadata(path):
    with h5py.File(path, 'r') as f:
        d = f[KEY]
        a = f['Acquisition'].attrs
        raw = f['Acquisition/Raw[0]'].attrs
        def decode(v):
            return v.decode() if isinstance(v, bytes) else str(v)
        dims = [decode(v) for v in d.attrs['Dimensions']]
        layout = 'nt_nc' if dims[0] == 'time' else 'nc_nt'
        nt, nc = d.shape if layout == 'nt_nc' else d.shape[::-1]
        return dict(path=str(Path(path).resolve()), bytes=Path(path).stat().st_size,
                    shape=list(d.shape), nt=nt, nc=nc, dtype=str(d.dtype), layout=layout,
                    fs=float(raw['OutputDataRate']), dx=float(a['SpatialSamplingInterval']),
                    start=decode(d.attrs['PartStartTime']), end=decode(d.attrs['PartEndTime']),
                    chunks=d.chunks, compression=d.compression)


def system_info():
    versions = {}
    for name in ['numpy', 'scipy', 'h5py', 'torch', 'psutil', 'dascore', 'daspy-toolbox', 'xdas', 'threadpoolctl']:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    def command(args):
        return subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace').stdout.strip()
    import torch
    return dict(platform=platform.platform(), python=sys.version, executable=sys.executable,
                cpu=platform.processor(), logical_cpus=os.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                ram_bytes=psutil.virtual_memory().total, versions=versions,
                gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                nvidia_smi=command(['nvidia-smi']),
                git_head=command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                git_status=command(['git', '-C', str(ROOT), 'status', '--short']),
                thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
                cache_policy='buffered OS IO; no cache purge; process restart is not cold cache')


class Monitor:
    """50 ms process memory sampling, plus OS-reported peak working set."""
    def __init__(self):
        self.rows = []
        self.stop_event = threading.Event()
        self.process = psutil.Process()

    def sample(self):
        m = self.process.memory_info()
        self.rows.append(dict(elapsed_s=time.perf_counter()-self.start_time,
                              rss_bytes=m.rss, private_bytes=getattr(m, 'private', 0)))

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.sample()
        def loop():
            while not self.stop_event.wait(.05):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.sample()

    def stats(self):
        m = self.process.memory_info()
        return dict(rss_peak_sampled_mb=max(r['rss_bytes'] for r in self.rows)/2**20,
                    private_peak_sampled_mb=max(r['private_bytes'] for r in self.rows)/2**20,
                    rss_baseline_mb=self.rows[0]['rss_bytes']/2**20,
                    os_peak_working_set_mb=getattr(m, 'peak_wset', m.rss)/2**20,
                    memory_sample_interval_s=.05)


def run_child(script, spec, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = {**spec, 'trial_name':name, 'trial_dir':str(directory.resolve()),
            'launch_time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    spec_path = directory / (name + '.config.json')
    write_json(spec_path, spec)
    log = directory / (name + '.log')
    started = time.perf_counter()
    with log.open('w', encoding='utf-8') as f:
        proc = subprocess.run([sys.executable, str(script), '--worker', str(spec_path)], stdout=f, stderr=subprocess.STDOUT)
    output = directory / (name + '.json')
    if proc.returncode or not output.exists():
        raise RuntimeError(f'{name} failed (exit {proc.returncode}); see {log}')
    result = json.loads(output.read_text(encoding='utf-8'))
    print(f'{name}: {result.get("total_s", result.get("compute_s", "done"))} s (process {time.perf_counter()-started:.1f}s)', flush=True)
    return result


def error_stats(output, reference):
    a = np.asarray(output, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    e = a-b
    return dict(max_abs_error=float(np.max(np.abs(e))),
                relative_l2_error=float(np.linalg.norm(e.ravel())/max(np.linalg.norm(b.ravel()),1e-30)),
                finite=bool(np.isfinite(a).all()))


def summary(rows, group_keys, value_keys):
    out=[]
    groups={}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    for group, items in groups.items():
        r=dict(zip(group_keys,group))
        for k in value_keys:
            v=np.array([x[k] for x in items if x.get(k) is not None], dtype=float)
            if v.size:
                r.update({k+'_mean':float(v.mean()),k+'_std':float(v.std(ddof=1)) if len(v)>1 else 0.,
                          k+'_min':float(v.min()),k+'_max':float(v.max()),k+'_n':len(v)})
        out.append(r)
    return out


def freeze(root, ds_dir, ooi_dir):
    """Write config.json/system.json for a new run.

    ``ds_dir``: folder of one-minute ZS HDF5 files; ``ooi_dir``: folder of OOI
    South Cable HDF5 files.
    """
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    ds_path=Path(ds_dir)
    ds=[]
    for p in sorted(ds_path.glob('*.h5')):
        m=metadata(p)
        if m['nt']==int(m['fs']*60):
            ds.append(m)
        if len(ds)==10:
            break
    ooi_path=Path(ooi_dir)
    ooi=[metadata(p) for p in sorted(ooi_path.glob('*.h5'))[:3]]
    config=dict(ds=ds,ooi=ooi,repeats=5,storage_repeats=3,
                fft=dict(fs=200.,fmin=15.,fmax=32.,taper_hz=0.,block_samples=2000,halo_samples=200),
                iir=dict(order=4,fs=200.,fmin=15.,fmax=32.,block_samples=200),
                compare_filter=dict(fs=1550.,fmin=1.,fmax=50.,order=4),
                native_dtype_note='FiLark CPU FFT/IIR internally float64; GPU FFT float32; report as-shipped precision')
    for files in [ds,ooi]:
        assert len({(m['nc'],m['fs'],m['layout']) for m in files})==1
    write_json(root/'config.json',config)
    write_json(root/'system.json',system_info())
    scripts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'benchmarks').glob('*.py')}
    write_json(root/'scripts_sha256.json',scripts)
    env='\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in md.distributions()))
    (root/'environment.txt').write_text(env,encoding='utf-8')
    return config
