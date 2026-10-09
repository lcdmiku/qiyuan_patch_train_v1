# tools/

Development and validation utilities.  None of these are needed at inference time; the
submission entry point is `run.py` and the package it uses is `patchmatch/`.

## Validation

| script | purpose |
|---|---|
| `make_synthetic_input.py` | Build a mock `/input_path` from a labelled split (info.json + images/ + patches/ + queries.json + truth.json), so `run.py` can be exercised end to end without the real evaluation set. |
| `score_pred.py` | Score a produced `pred.json` against that `truth.json` with the competition metric. Deliberately independent of `patchmatch/eval/harness.py`, so the two act as cross-checks on each other. |
| `env_report.py` | Report installed package versions and CUDA/device details. Run this first when a run behaves unexpectedly on a new machine. |

## Performance measurement

These exist because the workload is launch-bound rather than FLOP-bound, so the intuitive
"this should be fast" reasoning is unreliable and every claim in the docs is backed by a
measurement from one of these.

| script | purpose |
|---|---|
| `profile_query.py` | Per-stage cost of one query (decode, each cascade level, localisation). The tool that found the `ncc_map` whole-image box convolution costing 1.1 s. |
| `profile_ncc_map.py` | Cost of each piece of `ncc_map` on a refinement-sized window, by dtype. This is where the 90x float64 convolution penalty was measured. |
| `bench_unfold.py` | Correctness and speed of the tiled `unfold` + GEMM path against an explicit sliding-window reference and against the per-template convolution fallback. Its small image also exercises the final-tile clamping that was a real bug. |
| `bench_refine.py` | Accuracy and cost of full-resolution refinement by dtype. Measured 60/60 exact hits in float32, which is why refinement uses float32. |

## Typical session

```bash
python tools/env_report.py
python tools/make_synthetic_input.py --split val --out .local_input --retrieval 5 --grounding 15
python run.py --input .local_input --output .local_output
python tools/score_pred.py --input .local_input --pred .local_output/pred.json
```

Full-split scoring uses the package harness instead (it reports more, including the
localisation-path breakdown and per-query timing):

```bash
python -m patchmatch.eval.harness --data qiyuan_patch_train_v1 --split val --profile
```
