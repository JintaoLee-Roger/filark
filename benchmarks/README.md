# Benchmarks

Scripts used for the performance evaluation in the FiLark paper. Each script
writes its raw results (per-trial CSV/JSON and a `system.json` with hardware
and library versions) to a new output folder; `benchmarks/results/` is not
tracked. The DAS records themselves are not included in this repository.

| Script | Measures | Paper |
|---|---|---|
| `bench_memory.py` | Viewer memory while browsing a long multi-file record | Sec. 6.2 |
| `bench_render.py` | Streaming renderer cost vs. channel count and sampling rate | Sec. 6.3 |
| `bench_gaps.py` | Detection and display of timestamp gaps | Sec. 6.4 |
| `make_synthetic.py` | Synthetic multi-file records with injected gaps/overlaps | Sec. 6.4 |
| `bench_dsp.py` | Full-record and chunked CPU/GPU FFT band-pass, stateful CPU IIR | Sec. 6.5 |
| `bench_iir_copy.py` | IIR before/after removing a redundant float64 copy | Sec. 6.5 |
| `bench_compare.py` | Opening, window reads and filtering in FiLark, DASCore, DASPy, xdas | Sec. 6.6 |
| `bench_random_windows.py` | Randomized all-channel window reads, same request list per package | Sec. 6.6 |
| `bench_ops.py` | Common operators through each package's documented call | Sec. 6.6 |
| `bench_storage.py` | Throughput of one vs. two independent buffered readers | Sec. 7 |
| `report_revision.py` | Rebuilds tables and figures from a results folder | |

`bench_memory.py`, `bench_render.py` and `bench_gaps.py` take the data folder
on the command line (`--help` lists the options). The remaining scripts read a
frozen `config.json` in the output root, which lists the input files and
parameters; create one with `_revision.freeze(root, ds_dir, ooi_dir)` and then
run, for example:

```sh
python benchmarks/bench_dsp.py --root <output_root>
python benchmarks/report_revision.py --root <output_root>
```

GPU benchmarks need PyTorch with CUDA. The comparison benchmarks additionally
need DASCore, DASPy and xdas, and `threadpoolctl`/`psutil` for provenance.

Notes on interpretation:

- Reads are buffered OS reads; repeated runs may be served from the page
  cache, so throughput figures are not physical disk bandwidth.
- In the package comparison each package keeps its native precision and
  parallel defaults; output dtypes and numerical differences are reported
  alongside the times.
