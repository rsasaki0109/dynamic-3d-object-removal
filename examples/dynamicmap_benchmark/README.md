# DynamicMap_Benchmark adapter

Source of [KTH-RPL/DynamicMap_Benchmark PR #28](https://github.com/KTH-RPL/DynamicMap_Benchmark/pull/28)
(`methods/dor_numpy/` there mirrors this folder).

After `pip install git+https://github.com/rsasaki0109/dynamic-3d-object-removal.git`,
copy this folder to `methods/dor_numpy/` in a DynamicMap_Benchmark checkout and run:

```bash
python main.py --data_dir /path/to/00 --algorithm fusion      # highest accuracy
python main.py --data_dir /path/to/00 --algorithm range
python main.py --data_dir /path/to/00 --algorithm scan_ratio
python main.py --data_dir /path/to/00 --algorithm temporal
```

`fusion` is the slowest of the four: ~11 min on seq 00 and ~29 min on seq 05 with
the default `--fusion-workers 6`; the others run in a few minutes.

Each command writes `dor_<algorithm>_output.pcd` into `data_dir`. Score with the
benchmark's `export_eval_pcd` + `scripts/py/eval/evaluate_all.py` (add `dor_fusion`
etc. to its hard-coded `algorithms` list first), or run the self-contained
reproducer in this repo:

```bash
python3 scripts/run_dynamicmap_benchmark.py --sequences 00 05
```
