# Uplifting RF2 exact inference

The launcher runs RFantibody RF2 from upstream commit `3e5a5bb1cb98befde9fc049ae3ff58cabb491f67` with the P4 pair cleanup, deferred scoring, and unused inference head optimizations. Use Linux x86-64 with a compatible NVIDIA driver and Python 3.10. The upstream lockfile pins `torch==2.3.1+cu118` and `dgl==2.4.0+cu118`.

```bash
uv sync --locked --python 3.10
```

You can also use the Python executable from an existing compatible RFantibody environment in place of `.venv/bin/python` below.

Supply `RF2_ab.pt` from the upstream weights distribution. The P4 checkpoint SHA-256 is `cb38916458f5a9ecad0cf33182d816a836204b58896c59bd5a4ef475f63950e7`. [Upstream checkpoint download script](https://github.com/RosettaCommons/RFantibody/blob/3e5a5bb1cb98befde9fc049ae3ff58cabb491f67/include/download_weights.sh)

```bash
.venv/bin/python uplifting/run_exact.py \
  --quiver candidates.qv --weights /path/to/RF2_ab.pt \
  --output rf2-results.qv --seed 42 --gpu 0
```

For one RFantibody HLT PDB with the required REMARK annotations, use `--pdb candidate.pdb` in place of `--quiver candidates.qv`. Both input modes produce a standard Quiver with the selected structures and RF2 scores. Each output tag ends in `_best`. A PDB basename with whitespace uses underscores between words to form a valid Quiver tag. The output path must be new. Omit `--gpu` to use the first visible device under the current CUDA environment.

The launcher follows the P4 Route A protocol. Each input record starts a fresh process with the supplied seed, 10 recycles (11 forwards), a hotspot reveal proportion of `0.0`, and upstream FP32 precision settings. Quiver records run in input order. To reproduce a pristine stock comparison, use the upstream commit above with the same checkpoint and single-record inputs, seed, settings, software environment, and GPU. The native upstream multi-record command uses one process and advances its RNG across records.

P4 measured byte-identical RF2 Quiver output on its short and typical acceptance cases on the same RTX 4090D, with separate processes per candidate. The portable launcher was prepared from that protocol.

The `main` branch points to the pristine upstream commit. The `opt/exact` branch contains the source changes and this launcher. The [P4 benchmark summary](benchmarks/p4-summary.json) records the inputs' sizes, wall times, and the scope of the output comparison. Use `python uplifting/compare_exact.py --help` for same-GPU comparisons against the baseline.
