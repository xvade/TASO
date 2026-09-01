# TASO rule generator — local extensions

The generator (`generator.cc`) enumerates candidate graph-rewrite rules
("transfers") by DFS over small operator graphs, checks each pair of graphs for
equivalence on random inputs, and writes the surviving rewrites to
`graph_subst.pb`. This is upstream TASO; the notes below cover **only this
project's modifications** — see `../../../BUGS.md` and the top-level
`../../../PROGRESS.md` for the why, and `../../../PROBLEMATIC.md` for the parts
that resist testing.

## What we added

| Change | Where | Purpose |
|---|---|---|
| PWL op family | `-DPWL_FOCUS` build flag | Restrict the op set to the piecewise-linear family (add/mul/sub/max/min/relu) so depth-3/4 enumeration is fast and priceable. |
| min/max/sub ops | op tables + `include/xflow/ops.h` shim | The generator can now build `ewsub`/`ewmax`/`ewmin`, producing the PWL rewrite family the maxout/lattice nets need. |
| `GEN_MAX_DEPTH` | `-DGEN_MAX_DEPTH=N` (default 3) | Compile-time DFS depth. Not an env var. |
| Quotient-relaxation flags | `RELAX_SUBGRAPH/SUPERGRAPH/VARORDER/SUBST` env vars | Disable the four minimization stages that quotient away cost-neutral (AC) rewrite families, so a downstream pruner can rebuild a minimal-complete set. |
| `GEN_COMMUTE` | env var | Enumerate commutative-op operands in **both** orders so `max(a,b)` and `max(b,a)` are both built. |

## The four quotient-relaxation flags

TASO's generator normally minimizes the emitted set by dropping rewrites it
considers redundant. Four stages do this; each has an off-by-default relaxation
flag (`generator.cc:1243-1247`, read at `:1948-1952`):

- `RELAX_SUBGRAPH` — skip common-subgraph pruning.
- `RELAX_SUPERGRAPH` — skip common-supergraph pruning.
- `RELAX_VARORDER` — skip the canonical variable-ordering constraint.
- `RELAX_SUBST` — skip the renaming-dedup of transfers (`same_via_subst`). This
  is the "AC-blindness" filter: with it active, `(op a b)=>(op b a)` is
  quotiented against `(op a b)=>(op a b)` and dropped.

Unset (the default) reproduces original TASO behavior. Relaxing lets the
generator *emit* the associativity/commutativity families it used to suppress.

### ⚠ Flags are presence-checked, not value-checked

Every flag is read as `getenv("X") != nullptr`. **`GEN_COMMUTE=0` is still ON.**
The only way to turn a flag off is to leave it unset. This cost two invalid
experiment runs; the probe test below and the harness guard against silent
mis-toggling.

## `GEN_COMMUTE` and the 2 GB trap

`GEN_COMMUTE` (`generator.cc:1411-1417`) makes the operand loop for commutative
ops start at `k=0` (instead of `k=j+1`) with `k!=j`, so both operand orders are
built. This is what *creates* binary commutativity rules — `RELAX_SUBST` alone
does not (it only decides whether copies are *kept*).

**Run `GEN_COMMUTE=1` with `RELAX_SUBST` UNSET.** The substitution dedup then
collapses `GEN_COMMUTE`'s flood of renamed copies down to the canonical
`(ewmax a b)=>(ewmax b a)` / `(ewmin a b)=>(ewmin b a)` representatives — you
get the commutativity rules **without** the copy explosion.

Setting `GEN_COMMUTE=1` **and** `RELAX_SUBST=1` together keeps every swapped
copy: ~12× the transfers at depth 2, and at depth 3 it overflowed the single
serialized protobuf message's hard 2 GB ceiling (3.22 GB / 14.4 M transfers →
`RuleCollection exceeded maximum protobuf size of 2GB`), producing zero
downstream rules. See `../../../NNs/reassoc_results/GEN_COMMUTE_SUBST_PROBE.md`
for the full mechanism and the depth-2 measurement table.

## Building

Inside the container: `bash compile_pwl.sh` (regenerates the protobuf binding,
then compiles `generator_pwl` with `-DPWL_FOCUS`). For depth or full-op
variants, add `-DGEN_MAX_DEPTH=N` / drop `-DPWL_FOCUS`. The binary needs
`LD_LIBRARY_PATH` to include the conda libs (`libprotobuf.so.32`); `env -i`
strips it and a *stale* `graph_subst.pb` is copied silently.

## Test

`tests/test_flags_probe.sh` — a fast (seconds) depth-2 regression test that
compiles the PWL generator and pins the exact transfer/rule/commutativity counts
for the four flag configurations. It is the executable form of
`GEN_COMMUTE_SUBST_PROBE.md`; run it in the container:
```
apptainer exec --no-mount bind-paths tensat.sif bash taso/src/generator/tests/test_flags_probe.sh
```
