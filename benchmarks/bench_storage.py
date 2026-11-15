"""One versus two independent FiLark readers on the same ten HDD files."""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import _revision as R
import numpy as np
import psutil


def reader(files,start_event,ready,result_queue,worker_id):
    try:
        from filark.io.fileset import TapeFileSet
        t=time.perf_counter(); tape=TapeFileSet([m['path'] for m in files]); setup=time.perf_counter()-t
        ready.put(worker_id)
        start_event.wait()
        reads=[]; checksum=0.; total_bytes=0
        with R.Monitor() as mon:
            start=time.perf_counter()
            bt=int(files[0]['fs']*10)
            for a in range(0,tape.shape[0],bt):
                b=min(tape.shape[0],a+bt)
                t=time.perf_counter(); block=np.asarray(tape[a:b,:]); read_s=time.perf_counter()-t
                t=time.perf_counter(); checksum+=float(np.sum(block,dtype=np.float64)); consume_s=time.perf_counter()-t
                reads.append(dict(worker=worker_id,start_sample=a,end_sample=b,bytes=block.nbytes,read_s=read_s,consume_s=consume_s))
                total_bytes+=block.nbytes
                del block
            elapsed=time.perf_counter()-start
        tape.close()
        result_queue.put(dict(worker=worker_id,bytes=total_bytes,total_s=elapsed,setup_s=setup,
                              checksum=checksum,reads=reads,**mon.stats()))
    except BaseException:
        ready.put(dict(error=traceback.format_exc()))
        result_queue.put(dict(error=traceback.format_exc()))


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',required=True)
    args=ap.parse_args(); root=Path(args.root).resolve()
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    d=root/'storage'; d.mkdir(exist_ok=True)
    ctx=mp.get_context('spawn'); rows=[]
    for rep in range(cfg['storage_repeats']):
        for readers in ([1,2] if rep%2==0 else [2,1]):
            name=f'readers_{readers}_{rep}'; target=d/(name+'.json')
            if target.exists():
                record=json.loads(target.read_text(encoding='utf-8'))
            else:
                event=ctx.Event(); ready=ctx.Queue(); results=ctx.Queue()
                # Contiguous disjoint halves: exactly the same total samples, no duplicate reads.
                partitions=np.array_split(np.arange(len(cfg['ds'])),readers)
                children=[ctx.Process(target=reader,args=([cfg['ds'][int(i)] for i in partition],event,ready,results,j)) for j,partition in enumerate(partitions)]
                for p in children:p.start()
                for _ in children:
                    message=ready.get(timeout=120)
                    if isinstance(message,dict): raise RuntimeError(message)
                disks_before=psutil.disk_io_counters(perdisk=True)
                t=time.perf_counter(); event.set()
                workers=[results.get(timeout=3600) for _ in children]
                total_s=time.perf_counter()-t
                disks_after=psutil.disk_io_counters(perdisk=True)
                physical_reads={k:dict(read_bytes=v.read_bytes-disks_before[k].read_bytes,
                                       read_count=v.read_count-disks_before[k].read_count,
                                       read_time_ms=v.read_time-disks_before[k].read_time)
                                for k,v in disks_after.items() if k in disks_before}
                for p in children:p.join()
                if any('error' in w for w in workers):raise RuntimeError(workers)
                byte_count=sum(w['bytes'] for w in workers)
                expected=sum(m['nt']*m['nc']*np.dtype(m['dtype']).itemsize for m in cfg['ds'])
                assert byte_count==expected,(byte_count,expected)
                record=dict(readers=readers,rep=rep,total_s=total_s,bytes=byte_count,
                            throughput_mb_s=byte_count/total_s/1e6,checksum=sum(w['checksum'] for w in workers),
                            max_worker_s=max(w['total_s'] for w in workers),min_worker_s=min(w['total_s'] for w in workers),
                            sum_peak_rss_mb=sum(w['rss_peak_sampled_mb'] for w in workers),workers=workers,
                            system_physical_disk_reads=physical_reads,
                            cache='buffered, cache not purged; see run order; not a physical-disk bandwidth bound')
                R.write_json(target,record)
                R.csv_write(d/(name+'.reads.csv'),[r for w in workers for r in w['reads']])
            rows.append({k:v for k,v in record.items() if not isinstance(v,(dict,list))})
            R.csv_write(d/'trials.csv',rows)
            R.csv_write(d/'summary.csv',R.summary(rows,['readers'],['total_s','throughput_mb_s','max_worker_s','sum_peak_rss_mb']))
            print(f'{name}: {record["total_s"]:.3f}s, {record["throughput_mb_s"]:.1f} MB/s',flush=True)
    checks=[r['checksum'] for r in rows]
    R.write_json(d/'correctness.json',dict(equal_total_bytes=len({r['bytes'] for r in rows})==1,
                  checksum_relative_spread=(max(checks)-min(checks))/max(abs(checks[0]),1e-30)))


if __name__=='__main__':main()
