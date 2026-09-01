#!/bin/bash
# Regression test for the generator's GEN_COMMUTE / RELAX_SUBST flag interaction.
# The executable form of NNs/reassoc_results/GEN_COMMUTE_SUBST_PROBE.md.
#
# Compiles a depth-2 PWL-focus generator (seconds) and pins, for four flag
# configurations, the transfer/egg-rule/commutativity counts. Run in-container:
#   apptainer exec --no-mount bind-paths tensat.sif bash taso/src/generator/tests/test_flags_probe.sh
# Exits nonzero if any assertion fails. Plain asserts, no pytest.
#
# LOAD-BEARING INVARIANT (the scientific claim, robust to op-set changes):
#   binary commutativity rules emitted = 0 / 2 / 0 / many for
#   CANON / GEN_COMMUTE / RELAX_SUBST / GEN_COMMUTE+RELAX_SUBST.
#   - GEN_COMMUTE with the subst-dedup ACTIVE (RELAX_SUBST unset) yields EXACTLY
#     the two canonical survivors (ewmax a b)=>(ewmax b a), (ewmin a b)=>(ewmin b a).
#   - RELAX_SUBST alone emits ZERO commutativity (dedup-off only KEEPS copies;
#     GEN_COMMUTE is what BUILDS the swapped operand).
#   - Both together keep the whole copy-flood (many commut, huge transfer count) --
#     the multiplicative blow-up that overflowed the 2 GB protobuf ceiling at depth 3.
#
# EXACT COUNTS below are the current generator's deterministic depth-2 output
# (verified stable across runs). They supersede the smaller numbers in the .md,
# which were measured on an earlier generator state (fewer ops, buggy probe).
# If an intentional op-set change shifts them, update the EXPECT_* values AND
# confirm the 0/2/0/many commutativity invariant still holds.
set -uo pipefail
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
GEN="$REPO/taso/src/generator"
OUT="$REPO/NNs/reassoc_results"
PY=/opt/conda/bin/python3
export PATH=/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$OUT:$REPO/NNs"
T=$(mktemp -d)
PASS=0; FAIL=0
ok(){  echo "  PASS: $1"; PASS=$((PASS+1)); }
bad(){ echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
assert_eq(){ [ "$1" = "$2" ] && ok "$3 (=$1)" || bad "$3 (got $1, want $2)"; }
assert_ge(){ [ "$1" -ge "$2" ] && ok "$3 ($1 >= $2)" || bad "$3 (got $1, want >= $2)"; }

cd "$GEN"
protoc -I ../core --cpp_out=. ../core/rules.proto 2>/dev/null
protoc -I ../core --python_out="$OUT" ../core/rules.proto 2>/dev/null
echo "== compiling depth-2 PWL-focus generator =="
g++ generator.cc rules.pb.cc -o "$T/gen2" -I ../../include -I/opt/conda/include \
    -L/opt/conda/lib -lprotobuf -std=c++11 -pthread -O2 -DPWL_FOCUS -DGEN_MAX_DEPTH=2 \
    || { echo "  FAIL: compile"; exit 1; }

# 2-leaf ewmax/ewmin => 2-leaf ewmax/ewmin swap
CPAT='\(ew(max|min) \?[a-z_0-9]+ \?[a-z_0-9]+\)=>\(ew(max|min) \?[a-z_0-9]+ \?[a-z_0-9]+\)'
declare -A TR EG CM
run(){ local tag="$1"; shift
  ( cd "$T" && env "$@" RELAX_SUBGRAPH=1 RELAX_SUPERGRAPH=1 RELAX_VARORDER=1 "$T/gen2" >"$T/$tag.out" 2>&1 )
  cp "$T/graph_subst.pb" "$T/$tag.pb"
  $PY "$REPO/NNs/pb2egg.py" "$T/$tag.pb" "$T/$tag.egg" >/dev/null 2>&1
  TR[$tag]=$(grep -oE 'Generated [0-9]+ Transfers' "$T/$tag.out" | grep -oE '[0-9]+' | tail -1)
  EG[$tag]=$(grep -c '=>' "$T/$tag.egg")
  CM[$tag]=$(grep -cE "$CPAT" "$T/$tag.egg")
}
run CANON
run COMMUTE       GEN_COMMUTE=1
run SUBST         RELAX_SUBST=1
run COMMUTE_SUBST GEN_COMMUTE=1 RELAX_SUBST=1

echo "== commutativity invariant (0 / 2 / 0 / many) =="
assert_eq "${CM[CANON]}"         "0"  "CANON emits no commutativity"
assert_eq "${CM[COMMUTE]}"       "2"  "GEN_COMMUTE + subst-on emits exactly 2 (ewmax, ewmin)"
assert_eq "${CM[SUBST]}"         "0"  "RELAX_SUBST alone emits no commutativity"
assert_ge "${CM[COMMUTE_SUBST]}" "20" "GEN_COMMUTE + RELAX_SUBST keeps the copy-flood"

echo "== the 2 COMMUTE survivors are precisely the ewmax and ewmin swaps =="
smax=$(grep -cE '\(ewmax \?[a-z_0-9]+ \?[a-z_0-9]+\)=>\(ewmax \?[a-z_0-9]+ \?[a-z_0-9]+\)' "$T/COMMUTE.egg")
smin=$(grep -cE '\(ewmin \?[a-z_0-9]+ \?[a-z_0-9]+\)=>\(ewmin \?[a-z_0-9]+ \?[a-z_0-9]+\)' "$T/COMMUTE.egg")
assert_eq "$smax" "1" "ewmax commutativity present"
assert_eq "$smin" "1" "ewmin commutativity present"

echo "== flood factor: COMMUTE+SUBST transfers >> CANON (2 GB trap at depth 3) =="
assert_ge "$(( ${TR[COMMUTE_SUBST]} / ${TR[CANON]} ))" "10" "COMMUTE+SUBST >= 10x CANON transfers"

echo "== exact current depth-2 counts (deterministic snapshot) =="
assert_eq "${TR[CANON]}"         "38"    "CANON transfers"
assert_eq "${TR[COMMUTE]}"       "267"   "COMMUTE transfers"
assert_eq "${TR[SUBST]}"         "488"   "SUBST transfers"
assert_eq "${TR[COMMUTE_SUBST]}" "34524" "COMMUTE_SUBST transfers"
assert_eq "${EG[CANON]}"         "27"    "CANON egg rules"
assert_eq "${EG[COMMUTE]}"       "148"   "COMMUTE egg rules"
assert_eq "${EG[SUBST]}"         "383"   "SUBST egg rules"
assert_eq "${EG[COMMUTE_SUBST]}" "4293"  "COMMUTE_SUBST egg rules"

rm -rf "$T"
echo "======================================"
echo "GENERATOR PROBE TESTS: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo "ALL_TESTS_PASSED" || { echo "SOME_TESTS_FAILED"; exit 1; }
