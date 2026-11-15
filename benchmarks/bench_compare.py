"""Native four-framework comparison on ten full DS files (no data conversion)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import _revision as R
import h5py
import numpy as np
import pandas as pd


def ns(value):
    return pd.Timestamp(value).value


def canonical_parts(cfg, window):
    """Exact on-disk sample ranges defining a request; no padding or interpolation."""
    return [(cfg['ds'][p['file']],p['a'],p['b']) for p in window['parts']]


def freeze_windows(root,cfg):
    path=root/'compare_windows.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    windows=[]
    fs=cfg['ds'][0]['fs']
    # Eight interior requests and two requests straddling native file boundaries.
    for i in range(8):
        windows.append(dict(name=f'interior_{i}',parts=[dict(file=i,a=int(20*fs),b=int(30*fs))]))
    for i in [3,7]:
        a=cfg['ds'][i]; b=cfg['ds'][i+1]
        gap=(ns(b['start'])-ns(a['end']))/1e9-1/fs
        if abs(gap)>1/fs:
            raise ValueError(f'Unexpected real gap in cross-file request: {gap}')
        windows.append(dict(name=f'boundary_{i}',parts=[dict(file=i,a=a['nt']-int(5*fs),b=a['nt']),dict(file=i+1,a=0,b=int(5*fs))],timestamp_gap_s=gap))
    for w in windows:
        first,a,_=canonical_parts(cfg,w)[0]; last,_,b=canonical_parts(cfg,w)[-1]
        w['start_ns']=ns(first['start'])+round(a/first['fs']*1e9)
        w['last_ns']=ns(last['start'])+round((b-1)/last['fs']*1e9)
        w['n_samples']=sum(p['b']-p['a'] for p in w['parts'])
    R.write_json(path,windows)
    return windows


def open_native(software,cfg,directory,reuse=False):
    files=[m['path'] for m in cfg['ds']]
    if software=='filark':
        from filark.io.fileset import TapeFileSet
        obj=TapeFileSet(files)
        assert len(obj._files)==len(files)
        return obj
    if software=='daspy':
        import daspy
        # Read actual per-file headers rather than infer file timestamps.
        return daspy.Collection(files,meta_from_file='all')
    if software=='dascore':
        import dascore as dc
        from dascore.io.indexer import DirectoryIndexer
        from dascore.clients.dirspool import DirectorySpool
        index=Path(directory)/'index.h5'
        ix=DirectoryIndexer(Path(files[0]).parent,index_path=index)
        if not reuse:
            ix.update(paths=files,progress=None)
        obj=DirectorySpool(ix)
        assert len(obj.get_contents())==len(files), 'Never scan all 787 files'
        return obj
    if software=='xdas':
        import xdas as xd
        obj=xd.open(files,engine='prodml')
        assert obj.sizes['time']==sum(m['nt'] for m in cfg['ds'])
        # Native timestamps have microsecond resolution; strict zero tolerance
        # rejects fractional-rate slice endpoints after integer-timestamp rounding.
        # Public API: widen only the validation tolerance, keeping every tie point.
        coord=obj['time']
        fixed=coord.simplify(np.timedelta64(1,'us'),reduce=False)
        assert np.array_equal(coord.tie_indices,fixed.tie_indices)
        assert np.array_equal(coord.tie_values,fixed.tie_values)
        obj['time']=fixed
        return obj
    raise ValueError(software)


def read_native(software,obj,cfg,w):
    fs=cfg['ds'][0]['fs']; half=round(.49/fs*1e9)
    lo=np.datetime64(w['start_ns']-half,'ns'); hi=np.datetime64(w['last_ns']+half,'ns')
    parts=canonical_parts(cfg,w)
    if software=='filark':
        first,a,_=parts[0]
        offset=sum(m['nt'] for m in cfg['ds'][:w['parts'][0]['file']])+a
        return np.asarray(obj[offset:offset+w['n_samples'],:])
    if software=='daspy':
        from daspy import DASDateTime
        # select reads by UTC; 0 tolerance prevents gratuitous extra seconds.
        start=DASDateTime.fromtimestamp((w['start_ns']-half)/1e9).astimezone()
        # DASPy's tmax is exclusive (round-to-index); other two APIs use closed slices.
        end=DASDateTime.fromtimestamp((w['last_ns']+round(1e9/fs))/1e9).astimezone()
        result=obj.select(start=start,end=end,readsec=True,tolerance=0)
        return result.data.T
    if software=='dascore':
        patches=list(obj.select(time=(lo,hi)))
        return np.concatenate([p.transpose('time','distance').data for p in patches],axis=0)
    if software=='xdas':
        result=obj.sel(time=slice(lo,hi)).load()
        return np.asarray(result.transpose('time','distance').values)


def filter_native(software,x,cfg):
    fs=cfg['compare_filter']['fs']; dx=cfg['ds'][0]['dx']; nt,nc=x.shape
    # Objects and coordinate construction occur outside the filter timer.
    if software=='filark':
        from filark.dsp.temporal import iir_bandpass_time
        return lambda:iir_bandpass_time(x,fs=fs,fmin=1.,fmax=50.,order=4,zero_phase=False)
    if software=='daspy':
        import daspy
        obj=daspy.Section(x.T,dx=dx,fs=fs)
        return lambda:obj.bandpass(1.,50.,corners=4,zerophase=False).data.T
    if software=='dascore':
        import dascore as dc
        coords={'time':dc.get_coord(start=0.,step=1/fs,shape=(nt,),units='s'),
                'distance':dc.get_coord(start=0.,step=dx,shape=(nc,),units='m')}
        obj=dc.Patch(data=x,coords=coords,dims=('time','distance'))
        return lambda:obj.pass_filter(time=(1.,50.),corners=4,zerophase=False).data
    if software=='xdas':
        import xdas as xd
        obj=xd.DataArray(x,coords={'time':{'tie_indices':[0,nt-1],'tie_values':[0.,(nt-1)/fs]},
                                   'distance':{'tie_indices':[0,nc-1],'tie_values':[0.,(nc-1)*dx]}})
        def apply():
            return xd.atoms.IIRFilter(order=4,cutoff=(1.,50.),btype='bandpass',dim='time')(obj).values
        return apply


def worker(spec):
    cfg=spec['config']; software=spec['software']; task=spec['task']
    # Imports are setup, not data-open or compute performance.
    __import__(software)
    directory=Path(spec['trial_dir'])/spec['trial_name']
    directory.mkdir(exist_ok=True)
    rows=[]
    if task=='io':
        with R.Monitor() as mon:
            t=time.perf_counter(); obj=open_native(software,cfg,directory)
            elapsed=time.perf_counter()-t
            mem=mon.process.memory_info()
            rows.append(dict(task='open',software=software,rep=spec['rep'],total_s=elapsed,condition='build_metadata_or_index',
                             rss_after_open_mb=mem.rss/2**20,private_after_open_mb=getattr(mem,'private',0)/2**20,
                             index_bytes=sum(p.stat().st_size for p in directory.glob('*.h5'))))
            if software=='dascore':
                del obj
                t=time.perf_counter(); obj=open_native(software,cfg,directory,reuse=True)
                elapsed=time.perf_counter()-t
                mem=mon.process.memory_info()
                rows.append(dict(task='open_reuse',software=software,rep=spec['rep'],total_s=elapsed,condition='reuse_persistent_index',
                                 rss_after_open_mb=mem.rss/2**20,private_after_open_mb=getattr(mem,'private',0)/2**20,
                                 index_bytes=sum(p.stat().st_size for p in directory.glob('*.h5'))))
            channels=np.unique(np.linspace(0,cfg['ds'][0]['nc']-1,16,dtype=int))
            for w in spec['windows']:
                print('WINDOW',w['name'],flush=True)
                try:
                    t=time.perf_counter(); y=read_native(software,obj,cfg,w)
                except Exception:
                    R.write_json(directory/'partial_rows.json',rows)
                    raise
                elapsed=time.perf_counter()-t
                # Reference is fetched after the timed request; all software follows the same protocol.
                ref=[]
                for m,a,b in canonical_parts(cfg,w):
                    with h5py.File(m['path'],'r') as f:
                        ref.append(f[R.KEY][a:b,channels])
                reference=np.concatenate(ref)
                shape_ok=y.shape==(w['n_samples'],cfg['ds'][0]['nc'])
                err=R.error_stats(y[:,channels],reference) if shape_ok else dict(finite=False,relative_l2_error=None,max_abs_error=None)
                rows.append(dict(task='window',software=software,rep=spec['rep'],window=w['name'],total_s=elapsed,
                                 n_samples=y.shape[0],n_channels=y.shape[1],expected_samples=w['n_samples'],shape_ok=shape_ok,
                                 data_mb=y.nbytes/1e6,**err))
                del y
                if not shape_ok or not err['finite'] or err['relative_l2_error']>1e-6:
                    R.write_json(directory/'window_failure.json',rows)
                    raise RuntimeError(f'Window mismatch: {rows[-1]}')
            close=getattr(obj,'close',None)
            if close: close()
        memory=mon.stats()
    elif task=='filter':
        from scipy.signal import butter,sosfilt
        file=cfg['ds'][0]
        t=time.perf_counter()
        with h5py.File(file['path'],'r') as f:
            x=f[R.KEY][:]
        read_s=time.perf_counter()-t
        channels=np.unique(np.linspace(0,x.shape[1]-1,16,dtype=int))
        reference=sosfilt(butter(4,[1.,50.],fs=file['fs'],btype='bandpass',output='sos'),x[:,channels].astype(np.float64),axis=0)
        t=time.perf_counter(); apply=filter_native(software,x,cfg); prepare_s=time.perf_counter()-t
        with R.Monitor() as mon:
            t=time.perf_counter(); y=np.asarray(apply()); elapsed=time.perf_counter()-t
        err=R.error_stats(y[:,channels],reference)
        rows.append(dict(task='filter',software=software,rep=spec['rep'],compute_s=elapsed,total_s=elapsed,
                         read_s=read_s,prepare_s=prepare_s,msamples_s=x.size/elapsed/1e6,output_dtype=str(y.dtype),
                         n_samples=x.shape[0],n_channels=x.shape[1],**err))
        memory=mon.stats()
        if not err['finite']:
            raise RuntimeError('Nonfinite filter result')
    else:
        raise ValueError(task)
    for r in rows:
        r.update(memory)
    from threadpoolctl import threadpool_info
    R.csv_write(directory/'memory.csv',mon.rows)
    R.write_json(Path(spec['trial_dir'])/(spec['trial_name']+'.json'),dict(rows=rows,threadpools=threadpool_info()))


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root'); ap.add_argument('--worker')
    args=ap.parse_args()
    if args.worker:
        worker(json.loads(Path(args.worker).read_text(encoding='utf-8'))); return
    root=Path(args.root).resolve(); cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    windows=freeze_windows(root,cfg); d=root/'compare'; d.mkdir(exist_ok=True)
    rows=[]
    for rep in range(cfg['repeats']):
        software_order=['filark','daspy','dascore','xdas']
        software_order=software_order[rep%4:]+software_order[:rep%4]
        for task in ['io','filter']:
            for software in software_order:
                name=f'{task}_{software}_{rep}'; path=d/(name+'.json')
                if path.exists(): result=json.loads(path.read_text(encoding='utf-8'))
                else: result=R.run_child(__file__,dict(config=cfg,software=software,task=task,rep=rep,windows=windows),d,name)
                rows.extend(result['rows'])
                R.csv_write(d/'trials.csv',rows)
                R.csv_write(d/'summary.csv',R.summary(rows,['task','software'],['total_s','compute_s','msamples_s','rss_peak_sampled_mb','private_peak_sampled_mb','relative_l2_error']))


if __name__=='__main__':
    main()
