# `src/core/` — the op layer (shape inference + cost), spec

Backend-independent core of TASO: one `.cc` per operator class (plus `ops.cc`,
`substitution.cc`, `graph_to_trt.cc`). This is the **foundation everything else
depends on** — tensat's `TensorAnalysis` trusts the shapes these constructors
compute, so a wrong rule here is a wrong shape everywhere downstream. Part of the
fork's owned code (`../../../TASO_SUMMARY.md` §2–3). Verified against the source
2026-09-02.

## The op lifecycle contract

Each operator is a subclass of `OpBase` (declared in `include/taso/ops.h`, ~652–1050).
`Model::get_or_create_<op>(...)` (the factory, in each op's `.cc`) does three things,
in order, and **the constructor is where shape inference lives**:

1. **`<Op>::<Op>(Model*, inputs…, params…)`** — validates inputs with `assert`s and
   writes `numOutputs` and each `outputs[k]` (`.numDim`, `.dim[]`, `.stride[]`,
   `.split[]`). *This is the spec of the op's output shape.* No backend needed.
2. **`measure_<op>_cost(op)`** — sets `op->runtime` (backend-specific: real cuDNN timing,
   or the CPU analytic estimate — **force-zeroed in our build**, see `TASO_SUMMARY.md` §5).
3. The op is cached in the `Model`'s per-type map and returned as an `Op` handle.

`<Op>::collect_costs(exe_time, flops, mem_acc, num_kernels)` (also per `.cc`) reports the
op's FLOPs/memory for `Graph::total_cost` — **used only by `Graph::optimize`, which we
never call.** `map_output` / `unmap_output` / `forward` are the execution hooks (real in
cuDNN, stubbed on CPU — we don't execute).

`Tensor` (`ops.h`) carries `numDim`, `dim[]`, `stride[]`, and `split[]` (a `SplitInfo`
per axis that records where a `Concat`/`Split` boundary sits, so the optimizer can undo
them). `MAX_NUM_INPUTS` / `MAX_NUM_OUTPUTS` bound arity.

## Shape-inference spec — the ops our pipeline builds

Rules below are read straight from each constructor. NCHW layout throughout;
`PD_MODE_SAME = 0`, `PD_MODE_VALID = 1`.

| Op (`.cc`) | Output shape | Key asserts / notes |
|---|---|---|
| **Conv2D** | 4-D. `outC = weight.dim[0]`; `groups = inC / weight.dim[1]`; SAME → `outH=⌈inH/strideH⌉`; VALID → `⌊(inH−kH)/strideH⌋+1`. | `input.numDim==4`, `weight.numDim==4`, `inC % weightC == 0`. **Stride-1 SAME preserves H,W** — the fact `Iconv`/`Cpool` and the pool rules rely on. |
| **Pool2D** (Max/Avg) | 4-D, channels unchanged; H,W by the same SAME/VALID formula as Conv2D. | `input.numDim==4`; every corpus pool is 3×3 **stride-1 SAME** → output == input shape. |
| **Matmul** | Batched: `input[…,M,K] × weight[…,K,N] → [—,M,N]`. | `numDim` equal, batch dims equal, `input.dim[-1]==weight.dim[-2]`. 2-D is the FC case. |
| **Mul** (`OP_MUL`, scalar mul / tensat `smul`) | `= x` (the tensor operand's shape). | **`assert(y.numDim == 0)`** — 2nd operand must be a 0-D scalar (broadcast unimplemented). This is the gate tensat's applier checks before building `smul`. |
| **Element** (`EW_ADD/MUL/SUB/MAX/MIN`) | NumPy broadcast: `numDim = max`, `out.dim[i] = max(dim1,dim2)`. | `SUB/MAX/MIN` are the fork's added types (26/27/28). |
| **Activation** (Relu/Sigmoid/Tanh) | `= input` (shape-preserving). | — |
| **Concat** | `= inputs[0]`, with `dim[axis] = Σ inputs[j].dim[axis]`; per-axis `SplitInfo` combined. | `n ≤ MAX_NUM_INPUTS`; only `Concat` (binary) has a tensat applier arm (concat3/4/5 don't). |
| **Split** | `numOutputs = |sizes|`; each output = input with `dim[axis] = sizes[i]`; boundary recorded in `SplitInfo`. | The inverse of `Concat`; splits `dim[axis]`. |
| **Transpose** | `out.dim[i] = input.dim[perm[i]]`; strides recomputed. | **`assert(shuffle)`** — shuffle must be `true` (value-invariant; only strides change). tensat/reconstruct force it. |
| **Enlarge** | 4-D; channels from `w1`, spatial `H,W` from `w2` (zero-pad `w1`'s kernel to `w2`'s, centered). | `assert(w1.numDim==4 && w2.numDim==4)`, `w1` spatial ≤ `w2` spatial. Mirrors `reconstruct_generic.py::enlarge_np`. |
| **Constant** (`Iconv`/`Cpool`/`Imatmul`/`Iewmul`) | `MagicConst` — no standalone shape; supplied by the consuming op. | tensat models these as `DataKind::Const` markers, resolved by the consumer (`tensat/MODIFICATIONS.md`). |
| **Reshape** | `out.dim = params` (the target shape). | Total volume must match. |

## Invariants downstream relies on

- **Stride-1 SAME pooling/conv preserves H,W** — the whole `Cpool == poolavg` and
  `poolmax/poolavg` shape-shortcut story in tensat (`../../../PROBLEMATIC.md` #8) is a
  direct consequence of the Conv2D/Pool2D shape formula above.
- **`Mul` requires a 0-D scalar** — tensat's `smul` applier gates on `numDim==0` and
  declines otherwise, exactly matching `Mul`'s `assert`.
- **`Transpose` shuffle is always `true`** — the ctor asserts it; tensat's make/apply and
  `reconstruct_generic.py` pass `shuffle=True` because of this.
- **Shapes must be exact** — tensat reads these `dim[]`s through the FFI into
  `TensorAnalysis::make()`; there's no re-derivation, so a wrong ctor is a silent
  downstream shape bug.

## Ops present but outside our active path (upstream, documented by role)

`BatchNorm`, `Cast`, `ElementWiseUnary`, `MergeGConv`, `Pad`, `Reduce`, `Resize`,
`Shape`, `Slice`, `Squeeze`, `Unsqueeze`, `TopK`, `Where`, `NoOp` — full upstream op
classes with the same lifecycle, but nothing in our ingest→rewrite→reconstruct path
builds them, and tensat has no egg arity / applier for them. `graph_to_trt.cc` is the
TensorRT export path (unused; we export via ONNX). Left as-is; documented here so their
role is on record.

## Tests & gaps

- **Shape/lifecycle correctness** is exercised transitively: `NNs/tests/run_tests.sh`
  ingests the tracked `graph_subst.pb` and round-trips through tensat (which asserts on
  bad shapes), and `test_taso_importer.py` drives the Graph builders through the ONNX
  importer.
- **Gap:** there is no direct C++ unit test of an individual constructor's shape output —
  building the Graph needs the compiled runtime, which can't run in-repo
  (`../../../PROBLEMATIC.md` #5). A focused C++ test binary (link `libtaso_runtime`, build
  a one-op graph, assert `outputs[0].dim`) is the right future addition once a taso build
  runs in CI. Tracked in `../../../TASO_SUMMARY.md` §8.
