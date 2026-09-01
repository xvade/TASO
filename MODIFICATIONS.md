# TASO fork — local modifications

This is a fork of [yycdavid/taso](https://github.com/yycdavid/taso) with local
changes for the verifiability project. This file is the **index of the delta**
from upstream; the deep rationale for each fix lives in the top-level
`../BUGS.md`, and the generator changes have their own
[`src/generator/README.md`](src/generator/README.md).

Everything below is **project-authored**; upstream TASO's own docs are
`README.md` / `INSTALL.md` / `docs/`.

## Build & packaging

| Change | File(s) | Notes |
|---|---|---|
| CPU-only build path (`USE_CUDA=OFF`) with a synthetic cost model | `CMakeLists.txt`, `config.cmake`, `src/cpu/{ops_cpu,measure_cost_cpu,execution_stubs_cpu}.cc` | Lets stages 1–2 (structural rewriting, no real kernels) run without a GPU. The GPU build (`build_gpu`) still fails `Cuda failure 35` — see `../PROBLEMATIC.md`. |
| Configurable runtime lib dir | `python/setup.py` (`TASO_LIB_DIR`) | Link the Python ext against `build/` or `build_gpu/` without editing setup.py. |

## ONNX importer / exporter fixes (see BUGS.md for each)

| Fix | File(s) |
|---|---|
| Gemm importer silently dropped the bias input | `python/taso/__init__.py` |
| Importer didn't register `MatMul` (capitalized) | `python/taso/__init__.py` |
| Reshape handler missed `Constant`-node shape args | `python/taso/__init__.py` |
| `export_onnx()` listed initializers as formal graph inputs | `python/taso/__init__.py` |
| `export_onnx()` emitted an invalid ONNX op name for Matmul | `python/taso/__init__.py` |
| `Graph::get_operator_int_attr` returned garbage in Release builds | `src/core/ops.cc` |
| `export_op` had no cases for `EW_SUB`/`EW_MAX`/`EW_MIN` (types 26/27/28) → any extracted min/max/sub form crashed on export | `src/core/ops.cc` |
| `operator_attrs` / `input_weight_names` for Sub/Max/Min (ONNX export of min/max-lowered graphs) | `python/taso/__init__.py`, `python/taso/_cython/{CCore.pxd,core.pyx}` |
| Exposed `Graph::preprocess_weights()` to Python | `python/taso/__init__.py`, `_cython/core.pyx` |

## Rule generator (`src/generator/`)

Added min/max/sub ops, the `-DPWL_FOCUS` op-family restriction, the
`-DGEN_MAX_DEPTH` compile define, the four `RELAX_*` quotient-relaxation env
flags, and the `GEN_COMMUTE` flag. Full semantics, the presence-check gotcha,
and the 2 GB protobuf trap are documented in
[`src/generator/README.md`](src/generator/README.md), with a fast regression
test at `src/generator/tests/test_flags_probe.sh`.

## Tests

- Generator flag behavior: `src/generator/tests/test_flags_probe.sh`.
- The importer/exporter and `export_op` fixes are exercised end-to-end by the
  pipeline tests in `../NNs/tests/run_tests.sh` (pb2egg round-trips the tracked
  `graph_subst.pb`; reconstruct scripts exercise the ONNX export path).
