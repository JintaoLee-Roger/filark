"""Random all-channel 10-second reads; identical request lists across frameworks."""
from __future__ import annotations
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import shutil
import time

import _revision as R
from bench_compare import ns, canonical_parts, open_native, read_native
import h5py
import numpy as np
import pandas as pd
import psutil


def requests(cfg):
    rng=np.random.default_rng(cfg['seed'])
    fs=cfg['ds'][0]['fs']; length=round(cfg['window_seconds']*fs)
    bounds=np.cumsum([0]+[m['nt'] for m in cfg['ds']]).tolist()
    # Uniform starts over the whole logical record, without reusing exact starts.
    starts=rng.choice(bounds[-1]-length+1,size=cfg['repeats']*cfg['windows_per_round'],replace=False)
    rounds=[]
    for rep in range(cfg['repeats']):
        windows=[]
        values=starts[rep*cfg['windows_per_round']:(rep+1)*cfg['windows_per_round']].copy()
        rng.shuffle(values)
        for order,start in enumerate(values):
            start=int(start); end=start+length; pos=start; parts=[]
            while pos<end:
                i=bisect.bisect_right(bounds,pos)-1
                stop=min(end,bounds[i+1])
                parts.append(dict(file=i,a=pos-bounds[i],b=stop-bounds[i]))
                pos=stop
            first=cfg['ds'][parts[0]['file']];last=cfg['ds'][parts[-1]['file']]
            windows.append(dict(name=f'r{rep}_w{order}',round=rep,order=order,logical_start=start,
                parts=parts,n_samples=length,cross_file=len(parts)>1,
                start_ns=ns(first['start'])+round(parts[0]['a']/fs*1e9),
                last_ns=ns(last['start'])+round((parts[-1]['b']-1)/fs*1e9)))
        rounds.append(windows)
    assert len(set(int(s) for s in starts))==len(starts)
    return rounds


def worker(spec):
    cfg=spec['config'];software=spec['software'];rep=spec['rep']
    __import__(software)
    d=Path(spec['trial_dir'])/spec['trial_name'];d.mkdir(exist_ok=True)
    t=time.perf_counter();obj=open_native(software,cfg,d);setup_s=time.perf_counter()-t
    channels=np.unique(np.linspace(0,cfg['ds'][0]['nc']-1,16,dtype=int))
    rows=[]
    with R.Monitor() as mon:
        for w in spec['windows']:
            before=psutil.disk_io_counters(perdisk=True)
            t=time.perf_counter()
            if software=='filark':
                # UTC and sample-index selection refer to the same source samples.
                start=obj.time_to_sample(w['start_ns']/1e9)
                y=np.asarray(obj[start:start+w['n_samples'],:])
            else:
                y=read_native(software,obj,cfg,w)
            elapsed=time.perf_counter()-t
            after=psutil.disk_io_counters(perdisk=True)
            disk_reads={k:v.read_bytes-before[k].read_bytes for k,v in after.items() if k in before}
            refs=[]
            for m,a,b in canonical_parts(cfg,w):
                with h5py.File(m['path'],'r') as f:refs.append(f[R.KEY][a:b,channels])
            shape_ok=y.shape==(w['n_samples'],cfg['ds'][0]['nc'])
            err=R.error_stats(y[:,channels],np.concatenate(refs)) if shape_ok else dict(finite=False,relative_l2_error=None,max_abs_error=None)
            row=dict(software=software,rep=rep,order=w['order'],window=w['name'],logical_start=w['logical_start'],
                     cross_file=w['cross_file'],total_s=elapsed,data_mb=y.nbytes/1e6,shape_ok=shape_ok,
                     output_dtype=str(y.dtype),physical_disk_read_bytes=disk_reads,**err)
            rows.append(row)
            R.write_json(d/'requests.json',rows)
            if not shape_ok or not err['finite'] or err['relative_l2_error']>1e-6:
                raise RuntimeError(f'Window samples differ: {row}')
            del y
    close=getattr(obj,'close',None)
    if close:close()
    R.csv_write(d/'memory.csv',mon.rows)
    R.write_json(Path(spec['trial_dir'])/(spec['trial_name']+'.json'),dict(rows=rows,setup_s=setup_s,**mon.stats()))


def report(root,cfg,rows,rounds):
    df=pd.DataFrame(rows);assert len(df)==4*cfg['repeats']*cfg['windows_per_round']
    assert not df.duplicated(['software','rep','order']).any()
    assert df.shape_ok.eq(True).all() and df.relative_l2_error.eq(0.).all()
    flat=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]
    R.csv_write(root/'trials.csv',flat)
    by_round=[]
    for (software,rep),g in df.groupby(['software','rep']):
        expected=[w['logical_start'] for w in rounds[int(rep)]]
        assert g.sort_values('order').logical_start.tolist()==expected
        by_round.append(dict(software=software,rep=int(rep),n=len(g),median_ms=g.total_s.median()*1000,
                            mean_ms=g.total_s.mean()*1000,min_ms=g.total_s.min()*1000,max_ms=g.total_s.max()*1000))
    R.csv_write(root/'rounds.csv',by_round)
    summary=R.summary(by_round,['software'],['median_ms','mean_ms','min_ms','max_ms'])
    R.csv_write(root/'summary.csv',summary)
    validation=dict(requests=len(df),unique_random_starts=len(set(df.logical_start)),rounds=cfg['repeats'],
                    cross_file_windows=sum(w['cross_file'] for rr in rounds for w in rr),
                    same_requests_and_order_for_all_software=True,all_shapes_match=True,
                    max_sampled_relative_l2_error=float(df.relative_l2_error.max()),
                    note='16 uniformly spaced channels checked over each complete requested duration; no cache purge')
    R.write_json(root/'validation.json',validation)
    lines=['# Random-window reading comparison','',
           'Five rounds; ten newly randomized all-channel 10-second windows per round from the same ten DS files. Uniform random start samples without exact reuse; randomized request order; identical requests/order for all frameworks within a round. Seed: '+str(cfg['seed'])+'.','',
           '| Software | Round median (ms, mean ± sample SD) | Round mean (ms, mean ± sample SD) |',
           '|---|---:|---:|']
    for r in summary:
        lines.append(f'| {r["software"]} | {r["median_ms_mean"]:.2f} ± {r["median_ms_std"]:.2f} | {r["mean_ms_mean"]:.2f} ± {r["mean_ms_std"]:.2f} |')
    lines+=['','Each framework processes 50 random requests; 200 requests total. Sample/index and UTC selection refer to the same requested data and are not treated as different capabilities. All frameworks use native access; actual array materialization is timed, setup/imports and reference checks are not.',
            '', 'This measures random access under the actual buffered OS-cache state. Random does not imply cold cache. Physical-disk read-byte deltas are retained per request, and software execution order rotates by round. No full-directory scan, data conversion, or SSD copy.',
            '', 'Validation: '+json.dumps(validation), '',
            'The xdas native timestamp-rounding workaround uses public simplify(tolerance=1us, reduce=False); tie points remain unchanged. DASPy uses actual per-file metadata; DASCore indexes only the ten selected paths.',
            '', 'Raw data: trials.csv, rounds.csv, requests.json, runs/*/requests.json. Protocol: config.json. Code snapshot: code/.','']
    (root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.DataFrame(by_round);order=['filark','daspy','dascore','xdas']
    fig,ax=plt.subplots(figsize=(6.5,3.8),layout='constrained')
    groups=[table[table.software==s].median_ms for s in order]
    ax.bar(['FiLark','DASPy','DASCore','xdas'],[g.mean() for g in groups],yerr=[g.std() for g in groups],capsize=4,
           color=['#4477aa','#66ccee','#228833','#ccbb44'])
    ax.set_ylabel('Random-window latency (ms)');ax.set_title('10 s, all channels; mean of five round medians')
    ax.set_ylim(bottom=0);ax.spines[['top','right']].set_visible(False)
    fig.savefig(root/'random_windows.svg',bbox_inches='tight');fig.savefig(root/'random_windows.png',dpi=220,bbox_inches='tight');plt.close(fig)
    print(json.dumps(validation,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root');ap.add_argument('--source-config');ap.add_argument('--worker')
    args=ap.parse_args()
    if args.worker:worker(json.loads(Path(args.worker).read_text(encoding='utf-8')));return
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    original=json.loads(Path(args.source_config).read_text(encoding='utf-8'))
    cfg=dict(ds=original['ds'],repeats=5,windows_per_round=10,window_seconds=10,seed=20261103)
    R.write_json(root/'config.json',cfg)
    rounds=requests(cfg);R.write_json(root/'requests.json',rounds)
    R.write_json(root/'system.json',R.system_info())
    code=root/'code';code.mkdir(exist_ok=True)
    for name in ['bench_random_windows.py','bench_compare.py','_revision.py']:shutil.copy2(Path(__file__).with_name(name),code/name)
    R.write_json(code/'sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')})
    rows=[]
    for rep in range(cfg['repeats']):
        order=['filark','daspy','dascore','xdas'];order=order[rep%4:]+order[:rep%4]
        for software in order:
            name=f'{software}_{rep}';p=root/'runs'/(name+'.json')
            result=json.loads(p.read_text(encoding='utf-8')) if p.exists() else R.run_child(__file__,
                dict(config=cfg,windows=rounds[rep],software=software,rep=rep),root/'runs',name)
            rows.extend(result['rows'])
            R.csv_write(root/'trials.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows])
    report(root,cfg,rows,rounds)


if __name__=='__main__':main()
