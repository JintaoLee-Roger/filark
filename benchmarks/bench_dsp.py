"""Full OOI revision workload: shipped CPU/GPU FFT and exact stateful CPU IIR."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time

import _revision as R
import numpy as np


def worker(spec):
    from filark.io.fileset import TapeFileSet
    from filark.dsp.temporal import fft_bandpass_time, iir_filter_time_rt, build_iir_sos
    from filark.dsp.gpu.temporal import fft_bandpass_time_gpu
    import torch
    from threadpoolctl import threadpool_info
    cfg=spec['config']
    gpu=spec['device']=='cuda'
    if gpu:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    with R.Monitor() as mon:
        start=time.perf_counter()
        tape=TapeFileSet([m['path'] for m in cfg['ooi']])
        t=time.perf_counter()
        native=np.asarray(tape[:,:],dtype=np.float32)
        x=native.T if tape.dims=='nc_nt' else native
        read_s=time.perf_counter()-t
        tape.close()
        nt,nc=x.shape
        channels=np.unique(np.linspace(0,nc-1,16,dtype=int))
        probe_x=x[:,channels].copy()
        mode=spec['mode']
        fcfg={k:cfg['fft'][k] for k in ['fs','fmin','fmax','taper_hz']}
        fn=(lambda a:fft_bandpass_time_gpu(a,device='cuda',**fcfg)) if gpu else (lambda a:fft_bandpass_time(a,**fcfg))
        warmup_s=0.
        if gpu and not spec.get('warmup',False):
            # A separate process warms the device, but cannot warm this context's cuFFT plans.
            # Use the complete formal workload in this process before the timed computation.
            tw=time.perf_counter()
            if mode=='full_fft':
                warm_y=fn(x)
                del warm_y
            else:
                bt=cfg['fft']['block_samples']; halo=cfg['fft']['halo_samples']
                for a in range(0,nt,bt):
                    b=min(nt,a+bt)
                    warm_y=fn(x[max(0,a-halo):min(nt,b+halo)])
                    del warm_y
            torch.cuda.synchronize()
            warmup_s=time.perf_counter()-tw
            torch.cuda.reset_peak_memory_stats()
        if gpu:
            torch.cuda.synchronize()
        t=time.perf_counter()
        if mode=='full_fft':
            y=fn(x)
            probe_y=y[:,channels].copy()
            output_dtype=str(y.dtype)
            del y
        elif mode=='chunk_fft':
            bt=cfg['fft']['block_samples']; halo=cfg['fft']['halo_samples']
            probe_y=np.empty_like(probe_x)
            for a in range(0,nt,bt):
                b=min(nt,a+bt); ra=max(0,a-halo); rb=min(nt,b+halo)
                y=fn(x[ra:rb])
                probe_y[a:b]=y[a-ra:b-ra,channels]
                del y
            output_dtype='float32'
        elif mode=='iir':
            icfg=cfg['iir']
            sos=build_iir_sos(fs=icfg['fs'],kind='bandpass',fmin=icfg['fmin'],fmax=icfg['fmax'],order=icfg['order'])
            zi=None; bt=icfg['block_samples']; probe_y=np.empty_like(probe_x)
            for a in range(0,nt,bt):
                b=min(nt,a+bt)
                y,zi=iir_filter_time_rt(x[a:b],sos,zi=zi)
                probe_y[a:b]=y[:,channels]
                del y
            output_dtype='float32'
        else:
            raise ValueError(mode)
        if gpu:
            torch.cuda.synchronize()
        compute_s=time.perf_counter()-t
        total_s=time.perf_counter()-start-warmup_s
        # Correctness is outside the measured performance interval, using full-duration channels.
        if mode=='iir':
            reference,_=iir_filter_time_rt(probe_x,sos)
        elif mode=='full_fft':
            reference=fft_bandpass_time(probe_x,**fcfg)
        else:
            reference=np.empty_like(probe_x)
            for a in range(0,nt,bt):
                b=min(nt,a+bt); ra=max(0,a-halo); rb=min(nt,b+halo)
                reference[a:b]=fft_bandpass_time(probe_x[ra:rb],**fcfg)[a-ra:b-ra]
        err=R.error_stats(probe_y,reference)
    row=dict(mode=mode,device=spec['device'],rep=spec['rep'],warmup=spec.get('warmup',False),
             read_s=read_s,compute_s=compute_s,total_s=total_s,n_samples=nt*nc,
             warmup_s_excluded=warmup_s,warmup_in_process=bool(gpu and not spec.get('warmup',False)),
             compute_msamples_s=nt*nc/compute_s/1e6,end_to_end_msamples_s=nt*nc/total_s/1e6,
             shape=[nt,nc],input_layout='nc_nt',input_dtype='int32 -> float32',output_dtype=output_dtype,
             block_output_policy='consume output, retain 16 full-duration channels for correctness; no disk output',
             correctness_channels=channels.tolist(),**err,**mon.stats(),
             gpu_allocated_peak_mb=torch.cuda.max_memory_allocated()/2**20 if gpu else None,
             gpu_reserved_peak_mb=torch.cuda.max_memory_reserved()/2**20 if gpu else None,
             threadpools=threadpool_info())
    d=Path(spec['trial_dir']); name=spec['trial_name']
    R.csv_write(d/(name+'.memory.csv'),mon.rows)
    R.write_json(d/(name+'.json'),row)
    if not err['finite'] or err['relative_l2_error']>1e-4:
        raise RuntimeError(f'Correctness failed: {err}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root'); ap.add_argument('--worker')
    args=ap.parse_args()
    if args.worker:
        worker(json.loads(Path(args.worker).read_text(encoding='utf-8')))
        return
    root=Path(args.root).resolve(); cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    d=root/'dsp'; d.mkdir(exist_ok=True)
    rows=[]
    def launch(mode,device,rep,warmup=False):
        name=f'{mode}_{device}_{"warmup" if warmup else rep}'
        path=d/(name+'.json')
        if path.exists() and device=='cuda' and not warmup:
            old=json.loads(path.read_text(encoding='utf-8'))
            if not old.get('warmup_in_process',False):
                legacy=d/'device_warm_only_superseded'; legacy.mkdir(exist_ok=True)
                for suffix in ['.json','.config.json','.memory.csv','.log']:
                    p=d/(name+suffix)
                    if p.exists(): p.replace(legacy/p.name)
        if path.exists():
            row=json.loads(path.read_text(encoding='utf-8'))
        else:
            row=R.run_child(__file__,dict(config=cfg,mode=mode,device=device,rep=rep,warmup=warmup),d,name)
        return row
    for mode in ['full_fft','chunk_fft']:
        launch(mode,'cuda',-1,True)
    for rep in range(cfg['repeats']):
        tasks=[('full_fft','cpu'),('full_fft','cuda'),('chunk_fft','cpu'),('chunk_fft','cuda'),('iir','cpu')]
        if rep%2:
            tasks.reverse()
        for mode,device in tasks:
            rows.append(launch(mode,device,rep))
            flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
            R.csv_write(d/'trials.csv',flat)
            R.csv_write(d/'summary.csv',R.summary(flat,['mode','device'],['read_s','compute_s','total_s','compute_msamples_s','end_to_end_msamples_s','rss_peak_sampled_mb','private_peak_sampled_mb','gpu_allocated_peak_mb','relative_l2_error']))


if __name__=='__main__':
    main()
