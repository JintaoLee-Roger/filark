"""Rebuild revision tables, figures, and completion checks from raw results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import _revision as R
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
    root=Path(args.root).resolve(); cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    tables=root/'tables'; figures=root/'figures'; code=root/'code'
    for d in [tables,figures,code]:d.mkdir(exist_ok=True)
    for name in ['_revision.py','bench_dsp.py','bench_compare.py','bench_storage.py','report_revision.py']:
        shutil.copy2(Path(__file__).with_name(name),code/name)
    R.write_json(code/'sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')})
    dsp=pd.read_csv(root/'dsp/trials.csv'); comp=pd.read_csv(root/'compare/trials.csv'); storage=pd.read_csv(root/'storage/trials.csv')
    assert len(dsp)==25 and not dsp.duplicated(['mode','device','rep']).any()
    assert dsp.groupby(['mode','device']).size().eq(5).all()
    assert dsp.finite.all() and (dsp.relative_l2_error<1e-4).all()
    assert dsp.loc[dsp.device=='cuda','warmup_in_process'].eq(True).all()
    assert comp.groupby(['task','software']).size().loc['filter'].eq(5).all()
    assert comp.groupby(['task','software']).size().loc['open'].eq(5).all()
    expected_software={'filark','daspy','dascore','xdas'}
    assert set(comp.loc[comp.task=='filter','software'])==expected_software
    assert set(comp.loc[comp.task=='open','software'])==expected_software
    windows=comp[comp.task=='window'].copy()
    assert len(windows)==200 and windows.shape_ok.eq(True).all()
    assert windows.finite.eq(True).all() and windows.relative_l2_error.lt(1e-6).all()
    assert len(storage)==6 and storage.groupby('readers').size().eq(3).all()
    assert set(storage.readers)=={1,2}
    assert storage.bytes.nunique()==1
    assert np.isfinite(storage.checksum).all()
    validation=dict(dsp_trials=len(dsp),filter_trials=int((comp.task=='filter').sum()),
                    open_trials=int((comp.task=='open').sum()),window_requests=len(windows),storage_trials=len(storage),
                    correctness='16 uniformly spaced channels over full output duration; window sample counts checked for all requests',
                    max_dsp_relative_l2=float(dsp.relative_l2_error.max()),
                    max_window_relative_l2=float(windows.relative_l2_error.max()),
                    max_filter_relative_l2=float(comp.loc[comp.task=='filter','relative_l2_error'].max()),
                    window_access_pattern='fixed prescribed windows reused across rounds; not a random-access experiment',
                    paper_gpu_metric='author clarified computation excludes file loading; use compute_s, which includes host/device transfers inside the API',
                    mac='excluded at user request; final Mac results remain on Mac')
    R.write_json(root/'completion_checks.json',validation)
    R.write_json(root/'dsp/correctness.json',dict(reference='CPU full-duration transforms on 16 uniformly spaced channels; IIR reference has no temporal chunking',
                  trials=dsp[['mode','device','rep','relative_l2_error','max_abs_error','finite']].to_dict('records')))
    R.write_json(root/'compare/correctness.json',dict(reference='Source H5 samples for windows; float64 SciPy causal SOS reference for filters; 16 channels over each full task duration',
                  windows=windows[['software','rep','window','shape_ok','relative_l2_error','finite']].to_dict('records'),
                  filters=comp.loc[comp.task=='filter',['software','rep','relative_l2_error','max_abs_error','finite']].to_dict('records')))
    R.csv_write(tables/'open_memory.csv',comp.loc[comp.task.isin(['open','open_reuse']),
                ['software','task','rep','total_s','rss_after_open_mb','private_after_open_mb','index_bytes']].to_dict('records'))
    speed=[]
    for mode in ['full_fft','chunk_fft']:
        c=dsp[(dsp['mode']==mode)&(dsp.device=='cpu')];g=dsp[(dsp['mode']==mode)&(dsp.device=='cuda')]
        speed.append(dict(mode=mode,cpu_compute_mean_s=c.compute_s.mean(),gpu_compute_mean_s=g.compute_s.mean(),
                          compute_speedup=c.compute_s.mean()/g.compute_s.mean(),cpu_total_mean_s=c.total_s.mean(),
                          gpu_total_mean_s=g.total_s.mean(),end_to_end_speedup=c.total_s.mean()/g.total_s.mean()))
    R.csv_write(tables/'dsp_speedup.csv',speed)
    # Do not treat 50 requests as 50 independent repeated experimental rounds.
    round_rows=[]
    for (software,rep),group in windows.groupby(['software','rep']):
        round_rows.append(dict(software=software,rep=int(rep),median_s=group.total_s.median(),
                               phase='first_round' if rep==0 else 'repeat_round',
                               min_s=group.total_s.min(),max_s=group.total_s.max(),mean_s=group.total_s.mean()))
    R.csv_write(tables/'window_rounds.csv',round_rows)
    R.csv_write(tables/'window_summary.csv',R.summary(round_rows,['software','phase'],['median_s','min_s','max_s','mean_s']))
    for group in ['dsp','compare']:
        shutil.copy2(root/group/'summary.csv',tables/(group+'_summary.csv'))
    disk_map=json.loads((root/'disk_mapping.json').read_text(encoding='utf-8-sig'))
    disk_number=next(x['DiskNumber'] for x in disk_map if x['DriveLetter']=='F')
    storage_conditions=[]
    for r in storage.to_dict('records'):
        detail=json.loads((root/'storage'/f'readers_{int(r["readers"])}_{int(r["rep"])}.json').read_text(encoding='utf-8'))
        count=detail['system_physical_disk_reads'][f'PhysicalDrive{disk_number}']['read_bytes']
        storage_conditions.append({**r,'physical_read_bytes':count,
                                   'cache_condition':'physical_reads_observed' if count else 'no_physical_reads_observed'})
    R.csv_write(tables/'storage_conditions.csv',storage_conditions)
    R.csv_write(tables/'storage_summary.csv',R.summary(storage_conditions,['readers','cache_condition'],['total_s','throughput_mb_s','physical_read_bytes']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(figures/(name+'.svg'),bbox_inches='tight')
        fig.savefig(figures/(name+'.png'),dpi=220,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for j,metric in enumerate(['total_s','compute_s']):
        for k,device in enumerate(['cpu','cuda']):
            groups=[dsp[(dsp['mode']==mode)&(dsp.device==device)][metric] for mode in ['full_fft','chunk_fft']]
            axes[j].bar(np.arange(2)+(k-.5)*.34,[x.mean() for x in groups],width=.34,
                        yerr=[x.std() for x in groups],capsize=3,label=device.upper())
        axes[j].set_xticks([0,1],['Full FFT','Chunked FFT']);axes[j].set_ylabel('Time (s)')
        axes[j].set_title('Read + compute (supplementary)' if metric=='total_s' else 'Compute + GPU transfers')
        axes[j].legend(frameon=False)
    save(fig,'dsp_fft')
    order=['filark','daspy','dascore','xdas']; labels=['FiLark','DASPy','DASCore','xdas']
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for ax,task,title in zip(axes,['open','window','filter'],['Metadata / index initialization','Fixed-window cached retrieval','Causal Butterworth, one minute']):
        groups=[]
        for software in order:
            if task=='window':v=pd.DataFrame(round_rows).query('software == @software and rep > 0').median_s
            else:v=comp[(comp.task==task)&(comp.software==software)].total_s
            groups.append(v)
        ax.bar(labels,[v.mean() for v in groups],yerr=[v.std() for v in groups],capsize=3,color=['#4477aa','#66ccee','#228833','#ccbb44'])
        ax.set_title(title,fontsize=10);ax.set_ylabel('Time (s)');ax.tick_params(axis='x',rotation=20);ax.set_ylim(bottom=0)
    save(fig,'framework_comparison')
    fig,ax=plt.subplots(figsize=(5,3.7),layout='constrained')
    sd=pd.DataFrame(storage_conditions)
    conditions=[(1,'physical_reads_observed'),(1,'no_physical_reads_observed'),(2,'no_physical_reads_observed')]
    groups=[sd[(sd.readers==n)&(sd.cache_condition==condition)].throughput_mb_s for n,condition in conditions]
    ax.bar(['1 reader\nphysical reads','1 reader\ncached','2 readers\ncached'],[v.mean() for v in groups],
           yerr=[v.std() if len(v)>1 else 0. for v in groups],capsize=4,color=['#999999','#4477aa','#ee6677'])
    for i,v in enumerate(groups):ax.text(i,v.mean()+120,f'n={len(v)}',ha='center')
    ax.set_ylabel('Aggregate buffered-read throughput (MB/s)');ax.set_title('Observed OS-cache conditions; no cache purge')
    save(fig,'storage_concurrency')
    lines=['# Revision experiment results','',
           '**Author clarification:** the manuscript GPU comparison excludes loading files. Its corresponding recorded metric is compute_s (CPU computation or GPU API computation plus host/device transfers), not total_s. The ~1.95x/~1.93x read-plus-compute ratios must not be presented as contradicting the manuscript GPU figures. Computation-call speedups are 5.73x and 5.37x.','',
           '**Window-access review note:** this original experiment reuses ten fixed locations across five rounds and is diagnostic only. The replacement random-access experiment is completed and archived separately at ../random_windows_20261103/README.md. Do not mix these fixed-window records with the new random requests.','',
           '**Filter-comparison review note:** the four-framework filter timings characterize the selected native API paths, with unequal internal precision, output dtype and parallelism. They are not a controlled same-precision/same-worker-count comparison and must not be used as a general filtering-speed ranking. The author-recommended performance path still needs confirmation. See FILTER_COMPARISON_REVIEW.md.','',
           'All times are from formal full-size tasks. No full-directory scan or SSD copy was performed.',
           'Mac final-result organization is excluded at the author\'s request.','',
           '## DSP (five independent processes per configuration)','',
           '| Task | Device | Read (s) | Compute / transfers (s) | Total (s) | Compute Msamples/s |',
           '|---|---|---:|---:|---:|---:|']
    fmt=lambda v:f'{v.mean():.3f} ± {v.std():.3f}'
    for (mode,dev),g in dsp.groupby(['mode','device']):
        lines.append(f'| {mode} | {dev} | {fmt(g.read_s)} | {fmt(g.compute_s)} | {fmt(g.total_s)} | {fmt(g.compute_msamples_s)} |')
    lines+=['','| Task | Device | Process RSS peak (MiB) | Sampled private peak (MiB) | Torch allocated peak (MiB) |',
            '|---|---|---:|---:|---:|']
    for (mode,dev),g in dsp.groupby(['mode','device']):
        gpu=fmt(g.gpu_allocated_peak_mb) if dev=='cuda' else 'N/A'
        lines.append(f'| {mode} | {dev} | {fmt(g.rss_peak_sampled_mb)} | {fmt(g.private_peak_sampled_mb)} | {gpu} |')
    lines+=['','## Framework comparison (five runs / five rounds)','',
            '| Software | Metadata/index build (s) | Reuse index (s) | First window round median (s) | Repeat window round median (s) | Filter (s) | Filter relative L2 error |',
            '|---|---:|---:|---:|---:|---:|---:|']
    rounds=pd.DataFrame(round_rows)
    for software in order:
        pick=lambda task:comp[(comp.software==software)&(comp.task==task)]
        reuse=pick('open_reuse').total_s
        first=rounds[(rounds.software==software)&(rounds.rep==0)].median_s.iloc[0]
        repeated=rounds[(rounds.software==software)&(rounds.rep>0)].median_s
        lines.append(f'| {software} | {fmt(pick("open").total_s)} | {fmt(reuse) if len(reuse) else "N/A"} | {first:.3f} | {fmt(repeated)} | {fmt(pick("filter").total_s)} | {pick("filter").relative_l2_error.max():.3g} |')
    lines+=['','## Storage (three trials per reader count; separated by observed cache condition)','',
            '| Readers | F: physical reads | n | Wall time (s) | Aggregate MB/s |','|---|---|---:|---:|---:|']
    for (n,condition),g in sd.groupby(['readers','cache_condition']):
        wall=fmt(g.total_s) if len(g)>1 else f'{g.total_s.iloc[0]:.3f}'
        speed=fmt(g.throughput_mb_s) if len(g)>1 else f'{g.throughput_mb_s.iloc[0]:.3f}'
        lines.append(f'| {n} | {condition} | {len(g)} | {wall} | {speed} |')
    lines+=['','## Interpretation and reproducibility','',
            '- OOI: three original files, 47,500 channels × 36,000 samples, 200 Hz, on-disk channel × time.',
            '- DS: ten complete native files, time × channel, 6,455 channels at nominal 1,550 Hz; no bulk conversion.',
            '- FFT: 15–32 Hz rectangular mask; chunk core 10 s, one-second halo each side, crop halo (not overlap-add). No temporal taper.',
            '- IIR: four-pole Butterworth prototype (four SOS for bandpass), causal zero start; DSP uses one-second blocks with exact state carry.',
            '- The shipped CPU FFT/IIR uses float64 internal arithmetic and returns float32. GPU FFT uses float32. These are shipped-backend comparisons, not equal-precision kernel comparisons.',
            '- FiLark CPU FFT uses NumPy FFT without an explicit channel-parallel scheduler. Captured BLAS thread counts must not be interpreted as evidence of multithreaded FFT execution.',
            '- Each GPU measured process performs a full-workload warmup in its own context. Warmup is excluded from compute and total; total is elapsed read/setup/compute with this interval subtracted. GPU transfers back to host are included.',
            '- Whole FFT produces a full host array. Streaming FFT/IIR consume full-width output blocks and retain sixteen full-duration channels for correctness; no results are serialized during timing.',
            '- All raw process results, sampled process memory, GPU allocated/reserved peaks and threadpool details are retained. Windows peak-working-set and sampled private bytes have different meanings.',
            '- Framework filtering uses the same float32 time-major array. Each framework keeps its native precision and parallel defaults; output dtypes and numerical error are reported, not silently standardized.',
            '- DASCore native float32 SOS coefficients may have different rounding error from float64 coefficients at this low corner / sample-rate ratio; inspect correctness results before interpreting speed.',
            '- Requested source-sample ranges are frozen in compare_windows.json. Half-open DASPy and inclusive xdas/DASCore slices are adapted to request the same samples. Reference checks inspect sixteen channels for the complete requested duration.',
            '- xdas native time-coordinate zero tolerance rejects sliced microsecond timestamps at fractional sample intervals. The public simplify(tolerance=1 microsecond, reduce=False) API widens validation tolerance only; assertions check that every tie index and tie value is unchanged. No third-party code or raw data is modified.',
            '- Window latency measures data retrieval, not full interactive GUI navigation. First rounds are reported separately from the four repeat rounds. All five rounds remain in raw tables; software order rotates across rounds. No cache-cold equivalence is claimed for the first round across software.',
            '- DASCore builds an external index for only the ten specified files, and reuse is reported separately. DASPy reads all ten actual headers; xdas uses the native ProdML engine. Initial metadata work differs and is explicitly reported.',
            '- IO-session peak memory covers opening and window reads; use rss_after_open_mb/private_after_open_mb for post-open memory. Filter memory has its own measured phase.',
            '- Buffered OS reads are used throughout. Restarting processes does not clear the cache. Reference reads may populate the cache. No result is labelled cold-cache or claimed as a physical HDD bandwidth bound.',
            '- Storage compares one versus two independent readers using disjoint contiguous file groups, the same total bytes and full-block checksums. It does not characterize RAID or shared-handle thread concurrency.',
            '- Physical disk counters show 17.93 GB read on F: during the initial single-reader trial, and zero on F: for subsequent trials. The apparent ~5.7 GB/s two-reader throughput is cache-served, not HDD bandwidth. These runs establish buffered-reader behavior and caching effects; they do not establish an uncached HDD concurrency speedup.',
            '', 'Rebuild figures and tables:', '```powershell',
            f'python benchmarks/report_revision.py --root "{root}"', '```','',
            'Measurement entry points (use a fresh output root containing a copy of config.json for a new run; existing complete trials are resumed, not overwritten):',
            '```powershell',
            'python benchmarks/bench_dsp.py --root <output_root>',
            'python benchmarks/bench_compare.py --root <output_root>',
            'python benchmarks/bench_storage.py --root <output_root>',
            '```','',
            'See config.json, system.json, environment.txt, code/, raw per-trial JSON/CSV/logs, and completion_checks.json.']
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
