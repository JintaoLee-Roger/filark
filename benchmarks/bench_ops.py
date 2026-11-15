"""Common operators through each framework's documented call, plus direct library calls.

Each trial is a fresh process. A failing or hanging trial is recorded and the run continues.
Exploratory: definitions differ between frameworks (padding, tapers), see NOTES below.
"""
import argparse, datetime, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

FS, DX = 1550., 5.11247444152832
BP = dict(fmin=1., fmax=50., order=4)
V = (1000., 8000.)
NOTES = {
 'iir_causal': 'order-4 Butterworth 1-50 Hz, causal',
 'iir_zerophase': 'order-4 Butterworth 1-50 Hz, zero phase; DASPy filters forward then backward without padding',
 'detrend': 'linear, along time',
 'envelope': 'magnitude of the analytic signal along time',
 'stft': 'hann, 256 samples, 128 overlap; DASPy Section.spectrogram returns the channel-averaged power spectrogram',
 'fk_fan': 'keep apparent velocities 1000-8000 m/s; tapers and padding are each framework default; direct = FFT round trip without a mask',
 'fft_bandpass': '1-50 Hz frequency-domain bandpass; only FiLark documents this operator',
}
SOFTWARE = {
 'iir_causal': ['direct','filark','filark_gpu','daspy','dascore','xdas'],
 'iir_zerophase': ['direct','filark','daspy','dascore','xdas'],
 'detrend': ['direct','filark','daspy','dascore','xdas'],
 'envelope': ['direct','filark','daspy','dascore','xdas'],
 'stft': ['direct','filark','filark_gpu','daspy','dascore','xdas'],
 'fk_fan': ['direct','filark','filark_gpu','daspy','dascore'],
 'fft_bandpass': ['direct','filark','filark_gpu'],
}
PER_CHANNEL = {'iir_causal','iir_zerophase','detrend','envelope'}


def reference(op, x):
    from scipy import signal
    x = x.astype(np.float64)
    sos = signal.butter(BP['order'], [BP['fmin'], BP['fmax']], fs=FS, btype='bandpass', output='sos')
    if op == 'iir_causal': return signal.sosfilt(sos, x, axis=0)
    if op == 'iir_zerophase': return signal.sosfiltfilt(sos, x, axis=0)
    if op == 'detrend': return signal.detrend(x, axis=0)
    if op == 'envelope': return np.abs(signal.hilbert(x, axis=0))


def build(op, sw, x):
    """Everything except the returned call is setup and stays outside the timer."""
    nt, nc = x.shape
    f = dict(fs=FS, fmin=BP['fmin'], fmax=BP['fmax'])
    if sw == 'direct':
        from scipy import signal, fft
        sos = signal.butter(BP['order'], [BP['fmin'], BP['fmax']], fs=FS, btype='bandpass', output='sos')
        if op == 'iir_causal': return lambda: signal.sosfilt(sos, x, axis=0)
        if op == 'iir_zerophase': return lambda: signal.sosfiltfilt(sos, x, axis=0)
        if op == 'detrend': return lambda: signal.detrend(x, axis=0)
        if op == 'envelope': return lambda: np.abs(signal.hilbert(x, axis=0))
        if op == 'stft': return lambda: signal.stft(x, fs=FS, window='hann', nperseg=256, noverlap=128, axis=0)[2]
        if op == 'fk_fan': return lambda: fft.irfft2(fft.rfft2(x), s=x.shape)
        if op == 'fft_bandpass':
            freq = np.fft.rfftfreq(nt, 1/FS); mask = ((freq >= BP['fmin']) & (freq <= BP['fmax']))[:, None]
            return lambda: fft.irfft(fft.rfft(x, axis=0)*mask, n=nt, axis=0)
    if sw == 'filark':
        import filark.dsp.temporal as T, filark.dsp.spectral as S, filark.dsp.fk as K
        if op == 'iir_causal': return lambda: T.iir_bandpass_time(x, zero_phase=False, order=BP['order'], **f)
        if op == 'iir_zerophase': return lambda: T.iir_bandpass_time(x, zero_phase=True, order=BP['order'], **f)
        if op == 'detrend': return lambda: T.detrend_time(x, kind='linear')
        if op == 'envelope': return lambda: T.analytic_envelope(x)
        if op == 'stft': return lambda: S.stft_gather(x, fs=FS, nperseg=256, noverlap=128, window='hann')[2]
        if op == 'fk_fan': return lambda: K.fk_fan_filter(x, dt=1/FS, dx=DX, vmin=V[0], vmax=V[1])
        if op == 'fft_bandpass': return lambda: T.fft_bandpass_time(x, **f)
    if sw == 'filark_gpu':
        import filark.dsp.gpu.temporal as T, filark.dsp.gpu.spectral as S, filark.dsp.gpu.fk as K
        d = dict(device='cuda')
        if op == 'iir_causal': return lambda: T.iir_bandpass_time_rt_gpu(x, order=BP['order'], **f, **d)[0]
        if op == 'stft': return lambda: S.stft_gather_gpu(x, fs=FS, nperseg=256, noverlap=128, window='hann', **d)[2]
        if op == 'fk_fan': return lambda: K.fk_fan_filter_gpu(x, dt=1/FS, dx=DX, vmin=V[0], vmax=V[1], **d)
        if op == 'fft_bandpass': return lambda: T.fft_bandpass_time_gpu(x, **f, **d)
    if sw == 'daspy':
        import daspy
        s = daspy.Section(x.T, dx=DX, fs=FS)
        if op == 'iir_causal': return lambda: s.bandpass(1., 50., corners=4, zerophase=False).data.T
        if op == 'iir_zerophase': return lambda: s.bandpass(1., 50., corners=4, zerophase=True).data.T
        if op == 'detrend': return lambda: s.detrending().data.T
        if op == 'envelope': return lambda: s.envelope().data.T
        if op == 'stft': return lambda: s.spectrogram(nperseg=256, noverlap=128)
        if op == 'fk_fan': return lambda: s.fk_filter(mode='retain', vmin=V[0], vmax=V[1]).data.T
    if sw == 'dascore':
        import dascore as dc
        coords = {'time': dc.get_coord(start=0., step=1/FS, shape=(nt,), units='s'),
                  'distance': dc.get_coord(start=0., step=DX, shape=(nc,), units='m')}
        p = dc.Patch(data=x, coords=coords, dims=('time', 'distance'))
        if op == 'iir_causal': return lambda: p.pass_filter(time=(1., 50.), corners=4, zerophase=False).data
        if op == 'iir_zerophase': return lambda: p.pass_filter(time=(1., 50.), corners=4, zerophase=True).data
        if op == 'detrend': return lambda: p.detrend('time', type='linear').data
        if op == 'envelope': return lambda: p.envelope('time').data
        if op == 'stft': return lambda: p.stft(taper_window='hann', samples=True, overlap=128, time=256).data
        if op == 'fk_fan': return lambda: p.slope_filter(filt=[V[0]*.8, V[0], V[1], V[1]*1.25]).data
    if sw == 'xdas':
        import xdas as xd, xdas.signal as xs
        da = xd.DataArray(x, coords={'time': {'tie_indices': [0, nt-1], 'tie_values': [0., (nt-1)/FS]},
                                     'distance': {'tie_indices': [0, nc-1], 'tie_values': [0., (nc-1)*DX]}})
        if op == 'iir_causal': return lambda: xs.filter(da, (1., 50.), 'bandpass', corners=4, zerophase=False, dim='time').values
        if op == 'iir_zerophase': return lambda: xs.filter(da, (1., 50.), 'bandpass', corners=4, zerophase=True, dim='time').values
        if op == 'detrend': return lambda: xs.detrend(da, type='linear', dim='time').values
        if op == 'envelope': return lambda: np.abs(xs.hilbert(da, dim='time').values)
        if op == 'stft': return lambda: xs.stft(da, window='hann', nperseg=256, noverlap=128, dim={'time': 'frequency'}).values
    raise KeyError((op, sw))


def worker(spec):
    import _revision as R
    op, sw = spec['op'], spec['software']
    if spec.get('file'):
        import h5py
        t = time.perf_counter()
        with h5py.File(spec['file'], 'r') as f: x = f[R.KEY][:]
        read_s = time.perf_counter()-t
    else:
        x = (np.random.default_rng(0).normal(size=(spec['nt'], spec['nc']))).astype(np.float32); read_s = 0.
    probes = np.unique(np.linspace(0, x.shape[1]-1, 16, dtype=int))
    ref = reference(op, x[:, probes]) if op in PER_CHANNEL else None
    if sw != 'direct' and sw != 'filark_gpu': __import__(sw)
    gpu = sw == 'filark_gpu'
    if gpu:
        import torch
        build(op, sw, x[:2048])()  # untimed warm-up
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); call = build(op, sw, x); prepare_s = time.perf_counter()-t
    with R.Monitor() as mon:
        t = time.perf_counter(); y = call()
        if isinstance(y, tuple): y = y[0]  # DASPy spectrogram returns (data, f, t)
        if gpu: torch.cuda.synchronize()
        compute_s = time.perf_counter()-t
    y = np.asarray(y)
    if y.shape == x.shape[::-1] and x.shape[0] != x.shape[1]: y = y.T
    row = dict(op=op, software=sw, rep=spec['rep'], compute_s=compute_s, read_s=read_s, prepare_s=prepare_s,
               out_shape=str(y.shape), out_dtype=str(y.dtype), status='ok')
    if ref is not None and y.shape == x.shape:
        row.update(R.error_stats(y[:, probes], ref))
    row.update(mon.stats())
    if gpu: row['gpu_peak_mb'] = torch.cuda.max_memory_allocated()/2**20
    R.write_json(Path(spec['trial_dir'])/(spec['trial_name']+'.json'), dict(rows=[row]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root'); ap.add_argument('--worker'); ap.add_argument('--repo', required=True)
    ap.add_argument('--file'); ap.add_argument('--source-config'); ap.add_argument('--nt', type=int, default=3100); ap.add_argument('--nc', type=int, default=64)
    ap.add_argument('--repeats', type=int, default=5); ap.add_argument('--no-gpu', action='store_true'); ap.add_argument('--timeout', type=int, default=1800)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo)/'benchmarks'))
    if a.worker: return worker(json.loads(Path(a.worker).read_text(encoding='utf-8')))
    import _revision as R
    if a.source_config: a.file = json.loads(Path(a.source_config).read_text(encoding='utf-8'))['ds'][0]['path']
    root = Path(a.root); d = root/'trials'; d.mkdir(parents=True, exist_ok=True)
    R.write_json(root/'config.json', dict(vars(a), notes=NOTES, started=datetime.datetime.now().isoformat()))
    rows = []
    for rep in range(a.repeats):
        for op, sws in SOFTWARE.items():
            sws = [s for s in sws if not (a.no_gpu and s == 'filark_gpu')]
            sws = sws[rep % len(sws):]+sws[:rep % len(sws)]
            for sw in sws:
                name = f'{op}_{sw}_{rep}'; out = d/(name+'.json'); log = d/(name+'.log')
                spec = dict(op=op, software=sw, rep=rep, file=a.file, nt=a.nt, nc=a.nc, trial_dir=str(d), trial_name=name)
                sp = d/(name+'.config.json'); R.write_json(sp, spec)
                t = time.perf_counter(); status = 'ok'
                if not out.exists():
                    try:
                        with log.open('w', encoding='utf-8') as f:
                            rc = subprocess.run([sys.executable, __file__, '--repo', a.repo, '--worker', str(sp)], stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
                        status = 'ok' if rc == 0 and out.exists() else f'failed rc={rc}'
                    except subprocess.TimeoutExpired:
                        status = 'timeout'
                if status == 'ok':
                    rows.extend(json.loads(out.read_text(encoding='utf-8'))['rows'])
                else:
                    rows.append(dict(op=op, software=sw, rep=rep, status=status))
                print(f'{datetime.datetime.now():%H:%M:%S} {name}: {status} ({time.perf_counter()-t:.1f}s)', flush=True)
                R.csv_write(root/'trials.csv', rows)
    (root/'RUN_DONE.txt').write_text(datetime.datetime.now().isoformat(), encoding='utf-8')
    print('RUN_DONE', flush=True)


if __name__ == '__main__':
    main()
