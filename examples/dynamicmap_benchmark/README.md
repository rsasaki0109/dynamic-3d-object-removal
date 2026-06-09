# DynamicMap_Benchmark adapter

Upstream PR template for [KTH-RPL/DynamicMap_Benchmark](https://github.com/KTH-RPL/DynamicMap_Benchmark).

After `pip install dynamic-object-removal`, copy this folder to `methods/dor_numpy/` in a
DynamicMap_Benchmark checkout and run:

```bash
python main.py --data_dir /path/to/00 --algorithm range
python main.py --data_dir /path/to/00 --algorithm scan_ratio
python main.py --data_dir /path/to/00 --algorithm temporal
```

Each command writes `dor_<algorithm>_output.pcd`. Use the benchmark repo's
`export_eval_pcd` + `scripts/py/eval/evaluate_all.py` for SA/DA/AA/HA, or run the
self-contained reproducer in this repo:

```bash
python3 scripts/run_dynamicmap_benchmark.py --sequences 00
```
