"""Matched before/after benchmark of the IIR input-copy optimization."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time

import _revision as R
import h5py
import numpy as np


def worker(spec):
    import filark.dsp.temporal as T
    if spec['variant']=='before':
        # Run the actual pre-change function, not a hand-rewritten approximation.
        source=Path(spec['baseline_source']).read_text(encoding='utf-8')
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='iir_filter_time')
        namespace=dict(vars(T))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'baseline_temporal.py','exec'),namespace)
        fn=namespace['iir_filter_time']
    else:
        fn=T.iir_filter_time
    t=time.perf_counter()
    with h5py.File(spec['file'],'r') as f:x=f[R.KEY][:]
    read_s=time.perf_counter()-t
    with R.Monitor() as monitor:
        t=time.perf_counter()
        y=fn(x,fs=1550.,kind='bandpass',fmin=1.,fmax=50.,order=4,zero_phase=False)
        compute_s=time.perf_counter()-t
    memory=monitor.stats()
    # Hash every output element in a canonical order, after timing/memory probing.
    digest=hashlib.sha256(memoryview(y.ravel(order='F')).cast('B')).hexdigest()
    row=dict(variant=spec['variant'],rep=spec['rep'],compute_s=compute_s,read_s=read_s,
             shape=list(y.shape),input_dtype=str(x.dtype),output_dtype=str(y.dtype),
             sha256_column_major=digest,**memory)
    d=Path(spec['trial_dir']);name=spec['trial_name']
    R.write_json(d/(name+'.json'),row)
    R.csv_write(d/(name+'.memory.csv'),monitor.rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root');ap.add_argument('--source-config');ap.add_argument('--worker')
    args=ap.parse_args()
    if args.worker:
        worker(json.loads(Path(args.worker).read_text(encoding='utf-8')));return
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(args.source_config).read_text(encoding='utf-8'))
    baseline=root/'baseline_temporal.py'
    if not baseline.exists():
        source=subprocess.run(['git','-C',str(R.ROOT),'show','HEAD:filark/dsp/temporal.py'],capture_output=True,check=True).stdout
        baseline.write_bytes(source)
    R.write_json(root/'config.json',dict(file=cfg['ds'][0],repeats=5,baseline_source=str(baseline),
                   fs=1550.,fmin=1.,fmax=50.,order=4,zero_phase=False,
                   measurement='same full minute, dtype and single-call SciPy path; only redundant copies differ'))
    R.write_json(root/'system.json',R.system_info())
    patch=subprocess.run(['git','-C',str(R.ROOT),'diff','--','filark/dsp/temporal.py','filark/io/stream_processing.py'],capture_output=True,check=True).stdout
    (root/'optimization.patch').write_bytes(patch)
    import shutil
    for p in [Path(__file__),Path(__file__).with_name('_revision.py'),R.ROOT/'filark/dsp/temporal.py',R.ROOT/'filark/io/stream_processing.py']:
        shutil.copy2(p,root/p.name)
    rows=[]
    for rep in range(5):
        for variant in (['before','after'] if rep%2==0 else ['after','before']):
            name=f'{variant}_{rep}';file=root/(name+'.json')
            row=json.loads(file.read_text(encoding='utf-8')) if file.exists() else R.run_child(__file__,
                dict(file=cfg['ds'][0]['path'],variant=variant,rep=rep,baseline_source=str(baseline)),root,name)
            rows.append(row)
            R.csv_write(root/'trials.csv',[{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows])
    assert len({r['sha256_column_major'] for r in rows})==1, 'Full-output bits differ'
    summary=R.summary(rows,['variant'],['compute_s','rss_peak_sampled_mb','private_peak_sampled_mb','os_peak_working_set_mb'])
    R.csv_write(root/'summary.csv',summary)
    before=next(r for r in summary if r['variant']=='before');after=next(r for r in summary if r['variant']=='after')
    result=dict(full_output_bitwise_identical=True,shape=rows[0]['shape'],output_dtype=rows[0]['output_dtype'],
                output_sha256=rows[0]['sha256_column_major'],
                speedup=before['compute_s_mean']/after['compute_s_mean'],
                time_reduction_percent=100*(1-after['compute_s_mean']/before['compute_s_mean']),
                sampled_peak_rss_reduction_mib=before['rss_peak_sampled_mb_mean']-after['rss_peak_sampled_mb_mean'])
    R.write_json(root/'validation.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
